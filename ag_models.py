"""Схема валидации и обучение моделей (LightGBM, CatBoost, XGBoost) с ограничением по времени."""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ag_features import time_periods
from ag_metrics import fold_scores, oof_score, score
from ag_utils import log, log_exc


@dataclass
class FoldPlan:
    folds: list
    mode: str  # "cv": тест = среднее моделей фолдов; "refit": модель заново на всех данных (время)
    oof_mask: np.ndarray
    time_col: str | None = None
    desc: str = ""
    row_weight: np.ndarray | None = None  # веса строк train (похожесть на тест) — только в финальных моделях


def subset_plan(plan: FoldPlan, rows) -> FoldPlan:
    """Keep original time boundaries when evaluating a subsample of a large task."""
    rows = np.asarray(rows, dtype=int)
    folds = [(np.flatnonzero(np.isin(rows, tr)), np.flatnonzero(np.isin(rows, va)))
             for tr, va in plan.folds]
    if any(not len(tr) or not len(va) for tr, va in folds):
        raise ValueError("Subsample left an empty validation fold")
    return FoldPlan(folds=folds, mode=plan.mode, oof_mask=plan.oof_mask[rows],
                    time_col=plan.time_col, desc=plan.desc + " (subset, same boundaries)",
                    row_weight=None if plan.row_weight is None else plan.row_weight[rows])


def make_plan(X: pd.DataFrame, y: np.ndarray, task_type: str, time_col=None, n_folds=5, seed=42) -> FoldPlan:
    n = len(X)
    if time_col is not None:
        per = time_periods(X[time_col])
        uniq = np.sort(per.dropna().unique())
        folds = []
        max_time_folds = 3 if n <= 50_000 else 2
        for p in uniq[::-1]:
            tr = np.where(per.values < p)[0]
            va = np.where(per.values == p)[0]
            if len(tr) < 0.25 * n or len(va) < 50:
                continue
            folds.append((tr, va))
            if len(folds) == max_time_folds:
                break
        if folds:
            mask = np.zeros(n, bool)
            for _, va in folds:
                mask[va] = True
            desc = f"по времени ({time_col}): валидация на последних периодах " + ", ".join(
                f"{len(va)} строк" for _, va in folds)
            return FoldPlan(folds=folds, mode="refit", oof_mask=mask, time_col=time_col, desc=desc)
        log("временной сплит найден, но периодов мало — используем обычную кросс-валидацию")
    if n > 400_000:
        n_folds = 3
    if task_type != "regression":
        counts = np.bincount(y)
        min_cls = int(counts[counts > 0].min()) if (counts > 0).any() else 0
        if min_cls >= 2:
            k = int(min(n_folds, min_cls))
            splitter = StratifiedKFold(k, shuffle=True, random_state=seed)
            folds = list(splitter.split(np.zeros(n), y))
        else:
            folds = list(KFold(min(n_folds, max(2, n)), shuffle=True, random_state=seed).split(np.zeros(n)))
    else:
        folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n)))
    return FoldPlan(folds=folds, mode="cv", oof_mask=np.ones(n, bool), desc=f"{len(folds)}-fold CV")


# ---------------------------------------------------------------- цели и параметры


def lgb_objective(task_type, metric, n_classes, variant=None):
    if task_type == "binary":
        return {"objective": "binary", "metric": {"roc_auc": "auc", "average_precision": "average_precision"}.get(metric, "binary_logloss")}
    if task_type == "multiclass":
        return {"objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss"}
    eval_m = "l1" if metric in ("mae", "medae", "smape") else ("mape" if metric == "mape" else "rmse")
    if variant == "huber":
        return {"objective": "huber", "metric": eval_m}
    if variant == "fair":
        return {"objective": "fair", "metric": eval_m}
    if variant == "l2":
        return {"objective": "regression", "metric": eval_m}
    if metric in ("mae", "medae", "smape"):
        return {"objective": "l1", "metric": "l1"}
    if metric == "mape":
        return {"objective": "mape", "metric": "mape"}
    return {"objective": "regression", "metric": "rmse"}


DEFAULT_LGB = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
}


@dataclass
class ModelConfig:
    """Конфигурация эксперимента: что меняется относительно базовой модели."""

    lgb_params: dict = field(default_factory=dict)
    objective_variant: str | None = None  # None | huber | fair
    huber_alpha: float | None = None
    target_transform: str = "none"  # none | log1p
    class_weight_power: float = 0.0
    recency_weight: float = 0.0
    drop: list = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class FitResult:
    name: str
    oof: np.ndarray
    test: np.ndarray | None
    score: float
    fold_scores: list
    iters: list
    importance: pd.Series | None
    seconds: float
    config: dict = field(default_factory=dict)


def transform_target(y, how):
    if how == "log1p":
        return np.log1p(np.clip(y, 0, None))
    return y


def inverse_target(p, how):
    if how == "log1p":
        return np.expm1(p)
    return p


def sample_weights(y, task_type, cfg: ModelConfig, time_values=None, row_weight=None):
    w = np.ones(len(y), dtype=float)
    if row_weight is not None and len(row_weight) == len(y):
        w *= np.asarray(row_weight, dtype=float)
    if task_type != "regression" and cfg.class_weight_power:
        cnt = np.bincount(y).astype(float)
        cw = (cnt.max() / np.maximum(cnt, 1)) ** cfg.class_weight_power
        w *= cw[y]
    if time_values is not None and cfg.recency_weight:
        t = np.asarray(time_values, dtype=float)
        span = np.nanmax(t) - np.nanmin(t)
        if span > 0:
            rel = (t - np.nanmin(t)) / span
            w *= np.exp(cfg.recency_weight * np.nan_to_num(rel))
    return w


def _as_proba(p, task_type):
    if task_type == "binary":
        p = np.asarray(p).reshape(-1)
        return np.column_stack([1 - p, p])
    return p


class _TimeGuard:
    def __init__(self, seconds):
        self.deadline = time.time() + max(5.0, seconds)
        self.cut = False  # обучение какого-то фолда прервано по времени: модель недоучена

    def expired(self):
        return time.time() > self.deadline

    def stop_training(self) -> bool:
        """Для колбэков обучения: время вышло — обучение прерывается, модель помечается недоученной."""
        if self.expired():
            self.cut = True
            return True
        return False


# ---------------------------------------------------------------- LightGBM


def _sanitize_tree_frames(X: pd.DataFrame, Xt: pd.DataFrame | None):
    """Ensure no raw object/string columns remain for LightGBM/XGBoost (cast to shared Categorical)."""
    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
           and not isinstance(X[c].dtype, pd.CategoricalDtype)]
    if not bad:
        return X, Xt
    X = X.copy()
    Xt = Xt.copy() if Xt is not None else None
    for c in bad:
        sa = X[c].astype("string")
        sb = Xt[c].astype("string") if Xt is not None else None
        both = pd.concat([sa, sb], ignore_index=True) if sb is not None else sa
        cats = pd.CategoricalDtype(categories=list(both.dropna().unique()))
        X[c] = pd.Categorical(sa.astype(object), dtype=cats)
        if Xt is not None:
            Xt[c] = pd.Categorical(sb.astype(object), dtype=cats)
    return X, Xt


def _rmse_raw_log1p(preds, data):
    """RMSE на исходной шкале для LightGBM, обученной на log1p(таргета)."""
    err = inverse_target(preds, "log1p") - inverse_target(data.get_label(), "log1p")
    return "rmse_raw", float(np.sqrt(np.average(err ** 2, weights=data.get_weight()))), False


def fit_lgb(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float,
            lr: float | None = None, rounds: int = 5000, es: int = 100, seed: int = 0, predict_test: bool = True,
            name: str = "lgb", offset_col: str | None = None) -> FitResult:
    """offset_col (регрессия): модель учит поправку к значению этой колонки (например, прошлому таргету объекта);
    где значения нет — к медиане таргета."""
    import lightgbm as lgb

    t0 = time.time()
    guard = _TimeGuard(time_limit)
    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    params = dict(DEFAULT_LGB)
    params.update(lgb_objective(spec.task_type, spec.metric, n_classes, cfg.objective_variant))
    params.update(cfg.lgb_params or {})
    if lr:
        params["learning_rate"] = lr
    params.update({"num_threads": n_jobs, "seed": seed, "verbose": -1})
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    X, Xt = _sanitize_tree_frames(X[cols], Xt[cols] if Xt is not None else None)
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    if cfg.objective_variant == "huber" and "alpha" not in (cfg.lgb_params or {}):
        # порог Huber — в шкале, в которой учится модель (после log1p он другой)
        params["alpha"] = cfg.huber_alpha or float(np.std(yt) * 0.5)
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    feval = None
    if spec.task_type == "regression" and cfg.target_transform == "log1p" and spec.metric in ("rmse", "mse", "r2"):
        # модель учится на log1p, а метрика задачи — на исходной шкале: останавливаемся по ней, а не по RMSE
        # логарифмов (WiDS: поправка к прошлому значению 34.34 -> 34.18, без поправки 34.87 -> 34.81)
        params["metric"] = "None"
        feval = _rmse_raw_log1p

    def stopper(env):
        if guard.stop_training():
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    use_offset = offset_col is not None and spec.task_type == "regression" and offset_col in X.columns
    raw_off = raw_off_t = None
    if use_offset:
        raw_off = transform_target(X[offset_col].astype(float).values, cfg.target_transform)
        raw_off_t = transform_target(Xt[offset_col].astype(float).values, cfg.target_transform) if Xt is not None else None

    def offset(values, fill):
        return np.where(np.isfinite(values), values, fill) if use_offset else None

    shape = (len(X), n_classes) if spec.task_type != "regression" else (len(X),)
    oof = np.full(shape, np.nan)
    test = []
    iters, imps = [], []
    for k, (tr, va) in enumerate(plan.folds):
        fill = float(np.median(yt[tr])) if use_offset else None
        off = offset(raw_off, fill) if use_offset else None
        dtr = lgb.Dataset(X.iloc[tr], yt[tr], weight=w[tr], free_raw_data=True,
                          init_score=off[tr] if use_offset else None)
        dva = lgb.Dataset(X.iloc[va], yt[va], weight=w[va], reference=dtr,
                          init_score=off[va] if use_offset else None)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva], feval=feval,
                      callbacks=[lgb.early_stopping(es, verbose=False), stopper])
        bi = m.best_iteration or m.current_iteration()
        iters.append(bi)
        p = m.predict(X.iloc[va], num_iteration=bi) + (off[va] if use_offset else 0.0)
        oof[va] = _as_proba(p, spec.task_type) if spec.task_type != "regression" else inverse_target(p, cfg.target_transform)
        imps.append(pd.Series(m.feature_importance("gain"), index=cols))
        if predict_test and plan.mode == "cv" and Xt is not None:
            pt = m.predict(Xt, num_iteration=bi) + (offset(raw_off_t, fill) if use_offset else 0.0)
            pt = _as_proba(pt, spec.task_type) if spec.task_type != "regression" else inverse_target(pt, cfg.target_transform)
            test.append(pt)
        if guard.expired() and k < len(plan.folds) - 1:
            log(f"{name}: время вышло на фолде {k + 1}/{len(plan.folds)}")
            break
    if predict_test and plan.mode == "refit" and Xt is not None:
        if len(iters) == len(plan.folds) or name in ("base", "lgb_main"):
            n_round = refit_rounds(iters, plan, len(X))
            guard.deadline = max(guard.deadline, time.time() + max(45.0, 0.45 * time_limit))
            fill = float(np.median(yt)) if use_offset else None
            m = lgb.train(params, lgb.Dataset(X, yt, weight=w, init_score=offset(raw_off, fill) if use_offset else None),
                          n_round, callbacks=[stopper])
            if m.current_iteration() >= max(1, int(0.5 * n_round)):
                pt = m.predict(Xt) + (offset(raw_off_t, fill) if use_offset else 0.0)
                test = _as_proba(pt, spec.task_type) if spec.task_type != "regression" else inverse_target(pt, cfg.target_transform)
            else:
                test = None
        else:
            test = None
    return _finish(name, oof, test, plan, spec, y, iters, imps, t0, cfg, cut=guard.cut)


def refit_rounds(best_iters, plan: FoldPlan, n_total: int) -> int:
    """Итерации модели, дообучаемой на всех периодах: медиана по фолдам, умноженная на рост выборки.
    На слабосигнальных задачах (медиана <= 5) не раздуваем число деревьев искусственным минимумом 20."""
    if isinstance(best_iters, (list, tuple, np.ndarray)):
        vals = [int(v) for v in best_iters if v is not None and v > 0]
        med = float(np.median(vals)) if vals else 20.0
    else:
        med = float(best_iters or 20)
    growth = n_total / max(1, len(plan.folds[0][0]))
    base = med if med <= 5 else max(med, 20.0)
    return max(1, int(round(base * float(np.clip(growth, 1.1, 1.8)))))


def _finish(name, oof, test_parts, plan, spec, y, iters, imps, t0, cfg, cut: bool = False):
    done = ~np.isnan(oof).reshape(len(oof), -1).any(1)
    mask = plan.oof_mask & done
    s = oof_score(spec.metric, spec.task_type, y, oof, plan=plan, mask=mask) if mask.any() else float("nan")
    fs = fold_scores(spec.metric, spec.task_type, y, oof, [f for f in plan.folds if done[f[1]].all()])
    imp = sum(imps) / len(imps) if imps else None
    if isinstance(test_parts, list):
        test = sum(test_parts) / len(test_parts) if test_parts else None
    else:
        test = test_parts
    config = cfg.to_dict() if cfg else {}
    if cut:
        config["time_cut"] = True
    return FitResult(name=name, oof=oof, test=test, score=s, fold_scores=fs, iters=iters, importance=imp,
                     seconds=time.time() - t0, config=config)


# ---------------------------------------------------------------- CatBoost


def fit_cat(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, cat_cols, n_jobs: int, time_limit: float,
            lr: float = 0.06, depth: int = 6, rounds: int = 4000, es: int = 150, seed: int = 0,
            name: str = "cat", abort_if_slow: bool = True, loss_override: str | None = None,
            l2_leaf_reg: float | None = None) -> FitResult | None:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    t0 = time.time()
    guard = _TimeGuard(time_limit)
    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    cat_set = set(cat_cols)
    cats = [c for c in cols if c in cat_set or isinstance(X[c].dtype, pd.CategoricalDtype)
            or not pd.api.types.is_numeric_dtype(X[c])]

    def conv(df):
        df = df[cols].copy()
        for c in cats:
            df[c] = df[c].astype("string").fillna("__nan__").astype(str)
        return df

    Xc, Xtc = conv(X), (conv(Xt) if Xt is not None else None)
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    if spec.task_type == "regression":
        eval_m = {"mae": "MAE", "medae": "MAE", "smape": "MAE", "mape": "MAPE"}.get(spec.metric, "RMSE")
        loss = loss_override or eval_m
        model_cls, extra = CatBoostRegressor, {"loss_function": loss, "eval_metric": eval_m}
    elif spec.task_type == "binary":
        ev = {"roc_auc": "AUC", "average_precision": "PRAUC"}.get(spec.metric, "Logloss")
        model_cls, extra = CatBoostClassifier, {"loss_function": "Logloss", "eval_metric": ev}
    else:
        model_cls, extra = CatBoostClassifier, {"loss_function": "MultiClass", "eval_metric": "MultiClass"}
    if l2_leaf_reg is not None:
        extra["l2_leaf_reg"] = float(l2_leaf_reg)

    n_parts = len(plan.folds) + (1 if plan.mode == "refit" else 0)
    fold0 = {"deadline": None, "cut": False}

    class Stopper:
        def after_iteration(self, info):
            if fold0["deadline"] is not None and time.time() > fold0["deadline"]:
                fold0["cut"] = True
                return False
            return not guard.stop_training()

    def make(iters, use_es):
        p = dict(iterations=iters, learning_rate=lr, depth=depth, random_seed=seed, thread_count=n_jobs,
                 verbose=False, allow_writing_files=False, **extra)
        if len(cols) > 100 and spec.task_type != "multiclass":
            p["rsm"] = 0.75
        if len(cols) > 200 and spec.task_type != "multiclass":
            p["border_count"] = 128
        if use_es:
            p.update(od_type="Iter", od_wait=es, use_best_model=True)
        return model_cls(**p)

    shape = (len(X), n_classes) if spec.task_type != "regression" else (len(X),)
    oof = np.full(shape, np.nan)
    test = []
    iters, imps = [], []

    def pred(m, df):
        if spec.task_type == "regression":
            return inverse_target(m.predict(df), cfg.target_transform)
        return m.predict_proba(df)

    for k, (tr, va) in enumerate(plan.folds):
        # первый фолд должен уложиться в свою долю времени, иначе модель целиком не успеет
        fold0["deadline"] = time.time() + 1.3 * time_limit / n_parts if (abort_if_slow and k == 0) else None
        m = make(rounds, True)
        m.fit(Pool(Xc.iloc[tr], yt[tr], cat_features=cats, weight=w[tr]),
              eval_set=Pool(Xc.iloc[va], yt[va], cat_features=cats, weight=w[va]), callbacks=[Stopper()])
        if fold0["cut"]:
            log(f"{name}: первый фолд не уложился в {1.3 * time_limit / n_parts:.0f}s — модель пропущена")
            return None
        bi = int(m.get_best_iteration() or m.tree_count_)
        iters.append(bi)
        oof[va] = pred(m, Xc.iloc[va])
        imps.append(pd.Series(m.get_feature_importance(), index=cols))
        if plan.mode == "cv" and Xtc is not None:
            test.append(pred(m, Xtc))
        if abort_if_slow and k == 0 and _too_slow(t0, plan, time_limit, name):
            return None
        if guard.expired():
            log(f"{name}: время вышло на фолде {k + 1}/{len(plan.folds)}")
            if k < len(plan.folds) - 1:
                return _finish(name, oof, None, plan, spec, y, iters, imps, t0, cfg, cut=guard.cut)
    if plan.mode == "refit" and Xtc is not None:
        m = make(refit_rounds(iters, plan, len(X)), False)
        m.fit(Pool(Xc, yt, cat_features=cats, weight=w))
        test = pred(m, Xtc)
    return _finish(name, oof, test, plan, spec, y, iters, imps, t0, cfg, cut=guard.cut)


# ---------------------------------------------------------------- XGBoost


def fit_xgb(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float,
            lr: float = 0.05, depth: int = 7, rounds: int = 5000, es: int = 150, seed: int = 0,
            name: str = "xgb", abort_if_slow: bool = True, objective_override: str | None = None) -> FitResult | None:
    import xgboost as xgb

    t0 = time.time()
    guard = _TimeGuard(time_limit)
    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    X, Xt = _sanitize_tree_frames(X[cols], Xt[cols] if Xt is not None else None)
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    wide = len(cols) > 100
    params = {"tree_method": "hist", "eta": lr, "max_depth": depth, "subsample": 0.8,
              "colsample_bytree": 0.55 if wide else 0.7, "min_child_weight": 5 if wide else 3,
              "lambda": 3.0 if wide else 1.0, "nthread": n_jobs, "seed": seed, "max_cat_to_onehot": 1,
              "max_bin": 256}
    if spec.task_type == "regression":
        eval_m = "mae" if spec.metric in ("mae", "medae", "smape") else "rmse"
        if objective_override == "pseudohuber":
            slope = float(np.std(yt) * 0.5)
            if not np.isfinite(slope) or slope <= 1e-6:
                slope = 1.0
            params.update(objective="reg:pseudohubererror", eval_metric=eval_m, huber_slope=slope)
        elif objective_override == "l2":
            params.update(objective="reg:squarederror", eval_metric=eval_m)
        elif spec.metric in ("mae", "medae", "smape"):
            params.update(objective="reg:absoluteerror", eval_metric="mae")
        else:
            params.update(objective="reg:squarederror", eval_metric="rmse")
    elif spec.task_type == "binary":
        ev = {"roc_auc": "auc", "average_precision": "aucpr"}.get(spec.metric, "logloss")
        params.update(objective="binary:logistic", eval_metric=ev)
    else:
        params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")

    n_parts = len(plan.folds) + (1 if plan.mode == "refit" else 0)
    fold0 = {"deadline": None, "cut": False}

    class Stopper(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if fold0["deadline"] is not None and time.time() > fold0["deadline"]:
                fold0["cut"] = True
                return True
            return guard.stop_training()

    def dm(df, lab=None, wt=None):
        return xgb.DMatrix(df, label=lab, weight=wt, enable_categorical=True, nthread=n_jobs)

    def pred(b, d, it):
        p = b.predict(d, iteration_range=(0, it))
        if spec.task_type == "regression":
            return inverse_target(p, cfg.target_transform)
        return _as_proba(p, spec.task_type)

    shape = (len(X), n_classes) if spec.task_type != "regression" else (len(X),)
    oof = np.full(shape, np.nan)
    test = []
    dtest = dm(Xt) if Xt is not None else None
    iters, imps = [], []
    for k, (tr, va) in enumerate(plan.folds):
        fold0["deadline"] = time.time() + 1.3 * time_limit / n_parts if (abort_if_slow and k == 0) else None
        p_fold = dict(params)
        if spec.task_type == "binary":
            p_fold["base_score"] = float(np.clip(np.average(yt[tr], weights=w[tr]), 1e-4, 1.0 - 1e-4))
        elif spec.task_type == "regression" and (objective_override == "pseudohuber" or spec.metric in ("mae", "medae", "smape")):
            p_fold["base_score"] = float(np.median(yt[tr]))
        dtr, dva = dm(X.iloc[tr], yt[tr], w[tr]), dm(X.iloc[va], yt[va], w[va])
        b = xgb.train(p_fold, dtr, rounds, evals=[(dva, "valid")], early_stopping_rounds=es, verbose_eval=False,
                      callbacks=[Stopper()])
        if fold0["cut"]:
            log(f"{name}: первый фолд не уложился в {1.3 * time_limit / n_parts:.0f}s — модель пропущена")
            return None
        bi = int(getattr(b, "best_iteration", b.num_boosted_rounds() - 1)) + 1
        iters.append(bi)
        oof[va] = pred(b, dva, bi)
        sc = b.get_score(importance_type="gain")
        imps.append(pd.Series({c: sc.get(c, 0.0) for c in cols}))
        if plan.mode == "cv" and dtest is not None:
            test.append(pred(b, dtest, bi))
        if abort_if_slow and k == 0 and _too_slow(t0, plan, time_limit, name):
            return None
        if guard.expired():
            log(f"{name}: время вышло на фолде {k + 1}/{len(plan.folds)}")
            if k < len(plan.folds) - 1:
                return _finish(name, oof, None, plan, spec, y, iters, imps, t0, cfg, cut=guard.cut)
    if plan.mode == "refit" and dtest is not None:
        n_round = refit_rounds(iters, plan, len(X))
        p_refit = dict(params)
        if spec.task_type == "binary":
            p_refit["base_score"] = float(np.clip(np.average(yt, weights=w), 1e-4, 1.0 - 1e-4))
        elif spec.task_type == "regression" and (objective_override == "pseudohuber" or spec.metric in ("mae", "medae", "smape")):
            p_refit["base_score"] = float(np.median(yt))
        b = xgb.train(p_refit, dm(X, yt, w), n_round)
        test = pred(b, dtest, n_round)
    return _finish(name, oof, test, plan, spec, y, iters, imps, t0, cfg, cut=guard.cut)


# ---------------------------------------------------------------- ExtraTrees


def _matrix_pair(X: pd.DataFrame, Xt: pd.DataFrame | None):
    """Таблицы (X, Xt) -> float32 для sklearn с единым словарём кодирования категорий строго по X."""
    a_cols, b_cols = [], []
    for c in X.columns:
        sa = X[c]
        sb = Xt[c] if Xt is not None else None
        if isinstance(sa.dtype, pd.CategoricalDtype):
            a_cols.append(sa.cat.codes.to_numpy(np.float32))
            if sb is not None:
                b_cols.append(sb.cat.codes.to_numpy(np.float32))
        elif pd.api.types.is_numeric_dtype(sa):
            a_cols.append(sa.to_numpy(np.float32, na_value=np.nan))
            if sb is not None:
                b_cols.append(pd.to_numeric(sb, errors="coerce").to_numpy(np.float32, na_value=np.nan))
        else:
            codes, uniques = pd.factorize(sa.astype("string"), sort=False)
            a_cols.append(codes.astype(np.float32))
            if sb is not None:
                umap = {v: i for i, v in enumerate(uniques)}
                b_cols.append(sb.astype("string").map(umap).fillna(-1.0).to_numpy(np.float32))
    A = np.column_stack(a_cols) if a_cols else np.zeros((len(X), 1), np.float32)
    A = np.where(np.isfinite(A), A, -1e9).astype(np.float32)
    if Xt is None:
        return A, None
    B = np.column_stack(b_cols) if b_cols else np.zeros((len(Xt), 1), np.float32)
    B = np.where(np.isfinite(B), B, -1e9).astype(np.float32)
    return A, B


def fit_et(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float, seed: int = 0,
           name: str = "et", n_estimators: int = 200, top_cols: list | None = None,
           max_cols: int | None = None) -> FitResult | None:
    """ExtraTrees: другая индуктивная предвзятость, чем у бустинга, — полезна в смеси (на UNSW-NB15 под долями
    классов теста одна почти не хуже LightGBM, а четвертью в смеси даёт +0.005 macro-F1)."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    t0 = time.time()
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    if top_cols and max_cols and len(cols) > max_cols:
        chosen = [c for c in top_cols if c in cols][:max_cols]
        if len(chosen) >= 15:
            cols = chosen
    A, B = _matrix_pair(X[cols], Xt[cols] if Xt is not None else None)
    n = len(A)
    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    params = dict(n_estimators=n_estimators, min_samples_leaf=3 if n < 200_000 else 10, max_features=0.5,
                  n_jobs=n_jobs, random_state=seed)

    def make(n_fit: int):
        p = dict(params)
        if n_fit > 300_000:
            p.update(bootstrap=True, max_samples=300_000)  # память и время: деревья на подвыборках
        return ExtraTreesRegressor(**p) if spec.task_type == "regression" else ExtraTreesClassifier(**p)

    def pred(m, a):
        if spec.task_type == "regression":
            return inverse_target(m.predict(a), cfg.target_transform)
        p = np.zeros((len(a), n_classes))
        p[:, m.classes_.astype(int)] = m.predict_proba(a)
        return p

    shape = (n, n_classes) if spec.task_type != "regression" else (n,)
    oof = np.full(shape, np.nan)
    test, imps = [], []
    for k, (tr, va) in enumerate(plan.folds):
        m = make(len(tr)).fit(A[tr], yt[tr], sample_weight=w[tr])
        oof[va] = pred(m, A[va])
        imps.append(pd.Series(m.feature_importances_, index=cols))
        if plan.mode == "cv" and B is not None:
            test.append(pred(m, B))
        del m
        if k == 0 and _too_slow(t0, plan, time_limit, name):
            return None
    if plan.mode == "refit" and B is not None:
        test = pred(make(n).fit(A, yt, sample_weight=w), B)
    return _finish(name, oof, test, plan, spec, y, [n_estimators] * len(plan.folds), imps, t0, cfg)


# ---------------------------------------------------------------- Linear (Ridge / Huber / LogReg)


def fit_linear(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float,
               kind: str = "ridge", seed: int = 0, name: str | None = None,
               top_cols: list | None = None, max_cols: int = 90) -> FitResult | None:
    """Линейная/аддитивная модель (RidgeCV / LassoCV / HuberRegressor / LogisticRegression):
    быстрое и сильное дополнение к деревьям на гладких зависимостях и в ансамбле."""
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, LassoCV, LogisticRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    t0 = time.time()
    name = name or kind
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    if plan.time_col is not None and spec.task_type == "regression":
        # Исключаем монотонные координаты времени: линейная модель иначе экстраполирует тренд на будущие периоды
        non_mono = [c for c in cols if c != plan.time_col and not str(c).endswith(("__ord", "__year"))]
        if len(non_mono) >= 5:
            cols = non_mono
    if top_cols and len(cols) > max_cols:
        chosen = [c for c in top_cols if c in cols][:max_cols]
        if len(chosen) >= 10:
            cols = chosen
    if not cols:
        return None
    n = len(X)
    if kind in ("huber", "lasso") and n > 100_000:
        return None  # плотный L-BFGS / coordinate descent на >100k строк медленный
    cat_cols = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object or str(X[c].dtype) == "string"]
    num_cols = [c for c in cols if c not in cat_cols]
    X_in = X[cols].copy()
    Xt_in = Xt[cols].copy() if Xt is not None else None
    for c in cat_cols:
        X_in[c] = X_in[c].astype("string").fillna("__missing__").astype(str)
        if Xt_in is not None:
            Xt_in[c] = Xt_in[c].astype("string").fillna("__missing__").astype(str)
    for c in num_cols:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if Xt_in is not None:
            Xt_in[c] = pd.to_numeric(Xt_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    dense = kind in ("huber", "lasso")
    y_lo = float(np.nanmin(y)) if spec.task_type == "regression" else None
    y_hi = float(np.nanmax(y)) if spec.task_type == "regression" else None

    def build_model():
        transformers = []
        if num_cols:
            transformers.append(("num", make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(with_mean=dense),
            ), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(
                handle_unknown="ignore", min_frequency=10, max_categories=60, sparse_output=not dense,
            ), cat_cols))
        pre = ColumnTransformer(transformers, sparse_threshold=0.0 if dense else 0.3)
        if spec.task_type == "regression":
            if kind == "huber":
                reg = HuberRegressor(alpha=10.0, epsilon=1.2, max_iter=500, tol=1e-3)
            elif kind == "lasso":
                reg = LassoCV(cv=3, n_alphas=25, max_iter=500, tol=1e-3, random_state=seed, n_jobs=min(n_jobs, 4))
            else:
                reg = RidgeCV(alphas=np.logspace(-1, 4, 11))
            return make_pipeline(pre, TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        clf = LogisticRegression(C=0.1, max_iter=120, tol=1e-3, solver="lbfgs")
        return make_pipeline(pre, clf)

    def fit_one(df_tr, y_tr, w_tr):
        m = build_model()
        step_name = m.steps[-1][0]
        try:
            m.fit(df_tr, y_tr, **{f"{step_name}__sample_weight": w_tr})
        except TypeError:
            m.fit(df_tr, y_tr)
        return m

    def pred(m, df):
        if spec.task_type == "regression":
            p = inverse_target(m.predict(df), cfg.target_transform)
            return np.clip(p, y_lo, y_hi)
        p = np.zeros((len(df), n_classes))
        clf = m.steps[-1][1]
        p[:, clf.classes_.astype(int)] = m.predict_proba(df)
        return p

    shape = (n, n_classes) if spec.task_type != "regression" else (n,)
    oof = np.full(shape, np.nan)
    test = []
    for k, (tr, va) in enumerate(plan.folds):
        m = fit_one(X_in.iloc[tr], yt[tr], w[tr])
        oof[va] = pred(m, X_in.iloc[va])
        if plan.mode == "cv" and Xt_in is not None:
            test.append(pred(m, Xt_in))
        if k == 0 and _too_slow(t0, plan, time_limit, name):
            return None
    if plan.mode == "refit" and Xt_in is not None:
        m = fit_one(X_in, yt, w)
        test = pred(m, Xt_in)
    return _finish(name, oof, test, plan, spec, y, [1] * len(plan.folds), [], t0, cfg)


# ---------------------------------------------------------------- HistGradientBoosting & Tabular MLP


def fit_hgb(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float,
            lr: float = 0.05, max_leaf_nodes: int = 31, max_iter: int = 450, seed: int = 0,
            name: str = "hgb", top_cols: list | None = None, max_cols: int = 160) -> FitResult | None:
    """HistGradientBoosting (sklearn): быстрый независимый бустинг на CPU, нативная поддержка категорий и L1."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from ag_metrics import better

    t0 = time.time()
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    if top_cols and len(cols) > max_cols:
        chosen = [c for c in top_cols if c in cols][:max_cols]
        if len(chosen) >= 15:
            cols = chosen
    if not cols:
        return None
    X_in = X[cols].copy()
    Xt_in = Xt[cols].copy() if Xt is not None else None
    cat_mask = []
    for c in cols:
        if isinstance(X_in[c].dtype, pd.CategoricalDtype):
            if X_in[c].cat.categories.size > 255:
                X_in[c] = X_in[c].cat.codes.astype("float32")
                if Xt_in is not None:
                    Xt_in[c] = Xt_in[c].cat.codes.astype("float32")
                cat_mask.append(False)
            else:
                cat_mask.append(True)
        elif pd.api.types.is_numeric_dtype(X_in[c]):
            cat_mask.append(False)
        else:
            codes, uniques = pd.factorize(X_in[c].astype("string"), sort=False)
            X_in[c] = codes.astype("float32")
            if Xt_in is not None:
                umap = {v: i for i, v in enumerate(uniques)}
                Xt_in[c] = Xt_in[c].astype("string").map(umap).fillna(-1.0).astype("float32")
            cat_mask.append(False)
    n = len(X_in)
    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    if n > 100_000:
        max_iter = min(max_iter, 300)
    if spec.task_type == "multiclass" and n_classes > 5:
        max_iter = min(max_iter, 250)
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    cats_arg = cat_mask if any(cat_mask) else None

    def make(iters):
        common = dict(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, max_iter=iters,
                      min_samples_leaf=30 if n < 100_000 else 80, l2_regularization=1.0,
                      categorical_features=cats_arg, early_stopping=False, random_state=seed)
        if spec.task_type == "regression":
            loss = "absolute_error" if spec.metric in ("mae", "medae", "smape") else "squared_error"
            return HistGradientBoostingRegressor(loss=loss, **common)
        return HistGradientBoostingClassifier(**common)

    def pred(m, df):
        if spec.task_type == "regression":
            return inverse_target(m.predict(df), cfg.target_transform)
        p = np.zeros((len(df), n_classes))
        p[:, m.classes_.astype(int)] = m.predict_proba(df)
        return p

    shape = (n, n_classes) if spec.task_type != "regression" else (n,)
    oof = np.full(shape, np.nan)
    test, iters = [], []
    try:
        from threadpoolctl import threadpool_limits
        _omp_ctx = lambda: threadpool_limits(limits=max(1, min(8, n_jobs)))  # noqa: E731
    except Exception:
        from contextlib import nullcontext
        _omp_ctx = nullcontext
    with _omp_ctx():
        for k, (tr, va) in enumerate(plan.folds):
            if spec.task_type != "regression" and len(np.unique(yt[tr])) < 2:
                return None
            m = make(max_iter)
            m.fit(X_in.iloc[tr], yt[tr], sample_weight=w[tr])
            best_it, best_sc, best_p = m.n_iter_, None, None
            step = max(5, m.n_iter_ // 25)
            staged = m.staged_predict(X_in.iloc[va]) if spec.task_type == "regression" else m.staged_predict_proba(X_in.iloc[va])
            for idx_it, sp in enumerate(staged, 1):
                if idx_it % step != 0 and idx_it != m.n_iter_:
                    continue
                if spec.task_type == "regression":
                    cand = inverse_target(sp, cfg.target_transform)
                else:
                    cand = np.zeros((len(va), n_classes))
                    cand[:, m.classes_.astype(int)] = sp
                sc = score(spec.metric, spec.task_type, y[va], cand)
                if best_sc is None or better(sc, best_sc, spec.metric):
                    best_sc, best_it, best_p = sc, idx_it, cand
            iters.append(best_it)
            oof[va] = best_p if best_p is not None else pred(m, X_in.iloc[va])
            if plan.mode == "cv" and Xt_in is not None:
                if best_it >= m.n_iter_:
                    test.append(pred(m, Xt_in))
                else:
                    m_cv = make(best_it)
                    m_cv.fit(X_in.iloc[tr], yt[tr], sample_weight=w[tr])
                    test.append(pred(m_cv, Xt_in))
            if k == 0 and _too_slow(t0, plan, time_limit, name):
                return None
        if plan.mode == "refit" and Xt_in is not None:
            n_round = refit_rounds(iters, plan, len(X))
            m = make(min(max_iter, n_round))
            m.fit(X_in, yt, sample_weight=w)
            test = pred(m, Xt_in)
    return _finish(name, oof, test, plan, spec, y, iters, [], t0, cfg)


def fit_mlp(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float,
            seed: int = 0, name: str = "mlp", top_cols: list | None = None,
            max_cols: int = 80) -> FitResult | None:
    """Табличная нейросеть MLP (обучается строго с нуля на CPU): гауссизация числовых признаков
    (QuantileTransformer) + OHE категорий, другая индуктивная предвзятость для ансамбля и стекинга."""
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

    t0 = time.time()
    cols = [c for c in X.columns if c not in set(cfg.drop)]
    if plan.time_col is not None and spec.task_type == "regression":
        non_mono = [c for c in cols if c != plan.time_col and not str(c).endswith(("__ord", "__year"))]
        if len(non_mono) >= 5:
            cols = non_mono
    if top_cols and len(cols) > max_cols:
        chosen = [c for c in top_cols if c in cols][:max_cols]
        if len(chosen) >= 10:
            cols = chosen
    n = len(X)
    if not cols or n < 100 or n > 120_000:
        return None
    cat_cols = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object or str(X[c].dtype) == "string"]
    num_cols = [c for c in cols if c not in cat_cols]
    X_in = X[cols].copy()
    Xt_in = Xt[cols].copy() if Xt is not None else None
    for c in cat_cols:
        X_in[c] = X_in[c].astype("string").fillna("__missing__").astype(str)
        if Xt_in is not None:
            Xt_in[c] = Xt_in[c].astype("string").fillna("__missing__").astype(str)
    for c in num_cols:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if Xt_in is not None:
            Xt_in[c] = pd.to_numeric(Xt_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    n_classes = len(spec.classes) if spec.task_type != "regression" else 1
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    y_lo = float(np.nanmin(y)) if spec.task_type == "regression" else None
    y_hi = float(np.nanmax(y)) if spec.task_type == "regression" else None

    def build_model(n_fit: int):
        tf = []
        if num_cols:
            tf.append(("num", make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                QuantileTransformer(n_quantiles=max(10, min(500, n_fit // 2 if n_fit < 500 else max(50, n_fit // 10))),
                                    output_distribution="normal", random_state=seed),
            ), num_cols))
        if cat_cols:
            tf.append(("cat", OneHotEncoder(
                handle_unknown="ignore", min_frequency=10, max_categories=40, sparse_output=False,
            ), cat_cols))
        pre = ColumnTransformer(tf, sparse_threshold=0.0)
        bs = int(min(256, max(32, n_fit // 20)))
        common = dict(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-2, batch_size=bs,
                      learning_rate_init=1e-3, max_iter=80, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=10, random_state=seed)
        if spec.task_type == "regression":
            return make_pipeline(pre, TransformedTargetRegressor(regressor=MLPRegressor(**common),
                                                                 transformer=StandardScaler()))
        return make_pipeline(pre, MLPClassifier(**common))

    def pred(m, df):
        if spec.task_type == "regression":
            return np.clip(inverse_target(m.predict(df), cfg.target_transform), y_lo, y_hi)
        p = np.zeros((len(df), n_classes))
        clf = m.steps[-1][1]
        p[:, clf.classes_.astype(int)] = m.predict_proba(df)
        return p

    shape = (n, n_classes) if spec.task_type != "regression" else (n,)
    oof = np.full(shape, np.nan)
    test, iters = [], []
    for k, (tr, va) in enumerate(plan.folds):
        if spec.task_type != "regression" and len(np.unique(yt[tr])) < 2:
            return None
        m = build_model(len(tr)).fit(X_in.iloc[tr], yt[tr])
        oof[va] = pred(m, X_in.iloc[va])
        est = m.steps[-1][1]
        est = getattr(est, "regressor_", est)
        iters.append(int(getattr(est, "n_iter_", 1)))
        if plan.mode == "cv" and Xt_in is not None:
            test.append(pred(m, Xt_in))
        if k == 0 and _too_slow(t0, plan, time_limit, name):
            return None
    if plan.mode == "refit" and Xt_in is not None:
        m = build_model(n).fit(X_in, yt)
        test = pred(m, Xt_in)
    return _finish(name, oof, test, plan, spec, y, iters, [], t0, cfg)


def _too_slow(t0, plan, time_limit, name) -> bool:
    """После первого фолда: успеет ли модель целиком? Если нет — не тратим время впустую."""
    per = time.time() - t0
    need = per * (len(plan.folds) + (1 if plan.mode == "refit" else 0))
    if need > time_limit * 1.1:
        log(f"{name}: прогноз {need:.0f}s > лимита {time_limit:.0f}s — модель пропущена")
        return True
    return False


def safe_fit(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        log_exc(getattr(fn, "__name__", "fit"))
        return None


def adversarial_scores(X: pd.DataFrame, Xt: pd.DataFrame, n_jobs: int, max_rows: int = 200_000):
    """Adversarial validation: насколько train отличается от test.

    Возвращает OOF-вероятность «строка похожа на тест» для каждой строки train, ROC-AUC классификатора
    train/test (0.5 — неразличимы) и самые «сдвинутые» признаки.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    n = len(X)
    A = pd.concat([X, Xt], ignore_index=True)
    lab = np.r_[np.zeros(n), np.ones(len(Xt))]
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 50,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "num_threads": n_jobs, "seed": 0}
    rng = np.random.default_rng(0)
    pred = np.zeros(len(A))
    imp = pd.Series(0.0, index=A.columns)
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(len(A)), lab):
        if len(tr) > max_rows:
            tr = rng.choice(tr, max_rows, replace=False)
        m = lgb.train(params, lgb.Dataset(A.iloc[tr], lab[tr]), 200)
        pred[va] = m.predict(A.iloc[va])
        imp += pd.Series(m.feature_importance("gain"), index=A.columns)
    auc = float(roc_auc_score(lab, pred))
    top = list(imp.sort_values(ascending=False).index[:6])
    return pred[:n], auc, top
