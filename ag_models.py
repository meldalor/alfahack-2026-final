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
        if loss == "Huber":
            delta = float(np.std(yt) * 0.5)
            if not np.isfinite(delta) or delta <= 1e-4:
                delta = 1.0
            loss = f"Huber:delta={delta:.4f}"
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
        if len(cols) > 200 or (spec.task_type == "multiclass" and n_classes >= 6):
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
    mc_many = spec.task_type == "multiclass" and n_classes >= 10
    params = {"tree_method": "hist", "eta": lr, "max_depth": min(depth, 5) if mc_many else depth, "subsample": 0.8,
              "colsample_bytree": 0.5 if mc_many else (0.55 if wide else 0.7), "min_child_weight": 5 if wide else 3,
              "lambda": 3.0 if wide else 1.0, "nthread": n_jobs, "seed": seed, "max_cat_to_onehot": 1,
              "max_bin": 128 if mc_many else 256}
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
    n_est_eff = min(n_estimators, 120) if n > 150_000 else n_estimators
    params = dict(n_estimators=n_est_eff, min_samples_leaf=3 if n < 150_000 else 15, max_features=0.5,
                  n_jobs=n_jobs, random_state=seed)

    def make(n_fit: int):
        p = dict(params)
        if n_fit > 150_000:
            p.update(bootstrap=True, max_samples=150_000)  # память и время: деревья на подвыборках
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
    """HistGradientBoosting (sklearn): быстрый независимый бустинг на CPU, нативная поддержка категорий и L1.
    Обучается чанками через warm_start с ранней остановкой и строгим контролем time_limit (без двойного refit)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from ag_metrics import better

    t0 = time.time()
    guard = _TimeGuard(time_limit)
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
    if spec.task_type == "multiclass" and n_classes >= 6:
        max_iter = min(max_iter, 150 if (n_classes >= 10 and n > 30_000) else 200)
        lr = max(lr, 0.08)
    yt = transform_target(y, cfg.target_transform) if spec.task_type == "regression" else y
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight)
    cats_arg = cat_mask if any(cat_mask) else None
    n_parts = len(plan.folds) + (1 if plan.mode == "refit" else 0)

    def make(iters, warm: bool = True):
        common = dict(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, max_iter=iters,
                      min_samples_leaf=30 if n < 100_000 else 80, l2_regularization=1.0,
                      categorical_features=cats_arg, early_stopping=False, warm_start=warm, random_state=seed)
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

    chunk = 20 if (spec.task_type == "multiclass" and n_classes >= 6) else 30
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
            fold0_lim = t0 + 1.3 * time_limit / n_parts if k == 0 else None
            m = make(min(chunk, max_iter), warm=True)
            best_it, best_sc, best_p, best_tp = 0, None, None, None
            no_imp = 0
            cur_it = 0
            while cur_it < max_iter:
                cur_it = min(max_iter, cur_it + chunk)
                m.max_iter = cur_it
                m.fit(X_in.iloc[tr], yt[tr], sample_weight=w[tr])
                cand = pred(m, X_in.iloc[va])
                sc = score(spec.metric, spec.task_type, y[va], cand)
                if best_sc is None or better(sc, best_sc, spec.metric):
                    best_sc, best_it, best_p = sc, cur_it, cand
                    if plan.mode == "cv" and Xt_in is not None:
                        best_tp = pred(m, Xt_in)
                    no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= 3:
                    break
                if fold0_lim is not None and time.time() > fold0_lim:
                    log(f"{name}: первый фолд не уложился в {1.3 * time_limit / n_parts:.0f}s — модель пропущена")
                    return None
                if guard.stop_training():
                    break
            iters.append(max(best_it, chunk))
            oof[va] = best_p if best_p is not None else pred(m, X_in.iloc[va])
            if plan.mode == "cv" and Xt_in is not None:
                test.append(best_tp if best_tp is not None else pred(m, Xt_in))
            if k == 0 and _too_slow(t0, plan, time_limit, name):
                return None
            if guard.expired() and k < len(plan.folds) - 1:
                log(f"{name}: время вышло на фолде {k + 1}/{len(plan.folds)}")
                return _finish(name, oof, None, plan, spec, y, iters, [], t0, cfg, cut=guard.cut)
        if plan.mode == "refit" and Xt_in is not None:
            n_round = min(max_iter, refit_rounds(iters, plan, len(X)))
            m = make(n_round, warm=False)
            m.fit(X_in, yt, sample_weight=w)
            test = pred(m, Xt_in)
    return _finish(name, oof, test, plan, spec, y, iters, [], t0, cfg, cut=guard.cut)


class _NumpyTabularResNet:
    """Компактная табличная нейросеть ResNet/MLP на чистом NumPy/OpenBLAS с поддержкой
    L1, SmoothL1 (Huber), MSE, BCEWithLogits и CrossEntropy, LayerNorm, SiLU, Dropout, AdamW,
    bias-corrected EMA и Top-K SWA усреднения весов."""

    def __init__(self, d_in: int, d_out: int, loss_kind: str = "l1", hidden: int = 128,
                 h2: int = 64, dropout: float = 0.12, lr: float = 2.5e-3, wd: float = 1e-2,
                 huber_delta: float = 0.35, ema_decay: float = 0.985, topk_avg: int = 3,
                 seed: int = 0):
        self.loss_kind = loss_kind
        self.d_out = d_out
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.wd = float(wd)
        self.delta = float(huber_delta)
        self.ema_decay = float(ema_decay)
        self.topk_avg = int(topk_avg)
        self.rng = np.random.default_rng(seed)

        def _lin(a, b):
            lim = 1.0 / np.sqrt(max(a, 1))
            return self.rng.uniform(-lim, lim, size=(a, b)).astype(np.float32)

        def _bias(a, b):
            lim = 1.0 / np.sqrt(max(a, 1))
            return self.rng.uniform(-lim, lim, size=(b,)).astype(np.float32)

        self.p = {
            "W1": _lin(d_in, hidden), "b1": _bias(d_in, hidden),
            "g1": np.ones(hidden, np.float32), "be1": np.zeros(hidden, np.float32),
            "Wr": _lin(hidden, hidden), "br": _bias(hidden, hidden),
            "gr": np.ones(hidden, np.float32), "ber": np.zeros(hidden, np.float32),
            "W2": _lin(hidden, h2), "b2": _bias(hidden, h2),
            "g2": np.ones(h2, np.float32), "be2": np.zeros(h2, np.float32),
            "W3": _lin(h2, d_out), "b3": _bias(h2, d_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step_t = 0
        self.n_iter_ = 0

    @staticmethod
    def _ln_fwd(x, g, be):
        mu = x.mean(axis=1, keepdims=True)
        xc = x - mu
        inv = 1.0 / np.sqrt(np.mean(xc * xc, axis=1, keepdims=True) + 1e-5)
        xh = xc * inv
        return xh * g + be, (xh, inv, g)

    @staticmethod
    def _ln_bwd(dy, cache):
        xh, inv, g = cache
        dg = np.sum(dy * xh, axis=0)
        dbe = np.sum(dy, axis=0)
        dxh = dy * g
        dx = inv * (dxh - np.mean(dxh, axis=1, keepdims=True) - xh * np.mean(dxh * xh, axis=1, keepdims=True))
        return dx, dg, dbe

    @staticmethod
    def _silu_fwd(x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
        return x * s, (x, s)

    @staticmethod
    def _silu_bwd(dy, cache):
        x, s = cache
        return dy * (s * (1.0 + x * (1.0 - s)))

    def _forward(self, x: np.ndarray, train: bool = False):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]
        n1, c_n1 = self._ln_fwd(z1, p["g1"], p["be1"])
        a1, c_a1 = self._silu_fwd(n1)
        if train and self.dropout > 0:
            m1 = (self.rng.random(a1.shape, dtype=np.float32) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            a1d = a1 * m1
        else:
            m1, a1d = None, a1
        zr = a1d @ p["Wr"] + p["br"]
        nr, c_nr = self._ln_fwd(zr, p["gr"], p["ber"])
        ar, c_ar = self._silu_fwd(nr)
        if train and self.dropout > 0:
            mr = (self.rng.random(ar.shape, dtype=np.float32) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            h1 = a1d + ar * mr
        else:
            mr, h1 = None, a1d + ar
        z2 = h1 @ p["W2"] + p["b2"]
        n2, c_n2 = self._ln_fwd(z2, p["g2"], p["be2"])
        a2, c_a2 = self._silu_fwd(n2)
        if train and self.dropout > 0:
            m2 = (self.rng.random(a2.shape, dtype=np.float32) >= self.dropout * 0.7).astype(np.float32) / (1.0 - self.dropout * 0.7)
            a2d = a2 * m2
        else:
            m2, a2d = None, a2
        out = a2d @ p["W3"] + p["b3"]
        return out, (x, c_n1, c_a1, m1, a1d, c_nr, c_ar, mr, h1, c_n2, c_a2, m2, a2d)

    def _loss_grad(self, out: np.ndarray, y: np.ndarray, sw: np.ndarray):
        w = sw / max(float(sw.sum()), 1e-12)
        if self.loss_kind == "huber":
            diff = out[:, 0] - y
            ad = np.abs(diff)
            loss = float(np.sum(w * np.where(ad <= self.delta, 0.5 * diff * diff / self.delta, ad - 0.5 * self.delta)))
            g = np.where(ad <= self.delta, diff / self.delta, np.sign(diff)) * w
            return loss, g[:, None].astype(np.float32)
        if self.loss_kind == "l1":
            diff = out[:, 0] - y
            loss = float(np.sum(w * np.abs(diff)))
            g = np.sign(diff) * w
            return loss, g[:, None].astype(np.float32)
        if self.loss_kind == "mse":
            diff = out[:, 0] - y
            loss = float(np.sum(w * 0.5 * diff * diff))
            return loss, (diff * w)[:, None].astype(np.float32)
        if self.loss_kind == "bce":
            z = np.clip(out[:, 0], -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            loss = float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))
            return loss, ((p - y) * w)[:, None].astype(np.float32)
        z = out - np.max(out, axis=1, keepdims=True)
        ez = np.exp(np.clip(z, -30.0, 0.0))
        pr = ez / np.sum(ez, axis=1, keepdims=True)
        yi = y.astype(np.int64)
        loss = float(-np.sum(w * np.log(np.clip(pr[np.arange(len(yi)), yi], 1e-12, 1.0))))
        pr[np.arange(len(yi)), yi] -= 1.0
        return loss, (pr * w[:, None]).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, sw: np.ndarray, max_epochs: int = 55,
            batch_size: int = 256, patience: int = 10, guard: _TimeGuard | None = None,
            n_jobs: int = 4):
        try:
            from threadpoolctl import threadpool_limits
            ctx = threadpool_limits(limits=min(4, max(1, n_jobs)))
        except Exception:
            from contextlib import nullcontext
            ctx = nullcontext()

        n = len(X)
        idx = self.rng.permutation(n)
        n_va = max(32, min(int(0.1 * n), 5000))
        if n < 160:
            n_va = max(16, int(0.15 * n))
        va_i, tr_i = idx[:n_va], idx[n_va:]
        Xtr, ytr, wtr = X[tr_i], y[tr_i], sw[tr_i]
        Xva, yva, wva = X[va_i], y[va_i], sw[va_i]
        best_val = float("inf")
        top_ckpts = []
        ema_p = {k: v.copy() for k, v in self.p.items()} if self.ema_decay > 0 else None
        no_imp = 0

        with ctx:
            for ep in range(max_epochs):
                if guard is not None and guard.stop_training():
                    break
                warm = min(1.0, (ep + 1) / 3.0)
                cos_f = 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * ep / max(max_epochs, 1)))
                lr_ep = self.lr * warm * cos_f
                perm = self.rng.permutation(len(Xtr))
                for s in range(0, len(Xtr), batch_size):
                    bi = perm[s:s + batch_size]
                    out, cache = self._forward(Xtr[bi], train=True)
                    _, dout = self._loss_grad(out, ytr[bi], wtr[bi])
                    x, c_n1, c_a1, m1, a1d, c_nr, c_ar, mr, h1, c_n2, c_a2, m2, a2d = cache
                    grads = {}
                    grads["W3"] = a2d.T @ dout
                    grads["b3"] = dout.sum(axis=0)
                    da2d = dout @ self.p["W3"].T
                    da2 = da2d * m2 if m2 is not None else da2d
                    dn2 = self._silu_bwd(da2, c_a2)
                    dz2, grads["g2"], grads["be2"] = self._ln_bwd(dn2, c_n2)
                    grads["W2"] = h1.T @ dz2
                    grads["b2"] = dz2.sum(axis=0)
                    dh1 = dz2 @ self.p["W2"].T
                    dar = dh1 * mr if mr is not None else dh1
                    dnr = self._silu_bwd(dar, c_ar)
                    dzr, grads["gr"], grads["ber"] = self._ln_bwd(dnr, c_nr)
                    grads["Wr"] = a1d.T @ dzr
                    grads["br"] = dzr.sum(axis=0)
                    da1d = dh1 + dzr @ self.p["Wr"].T
                    da1 = da1d * m1 if m1 is not None else da1d
                    dn1 = self._silu_bwd(da1, c_a1)
                    dz1, grads["g1"], grads["be1"] = self._ln_bwd(dn1, c_n1)
                    grads["W1"] = x.T @ dz1
                    grads["b1"] = dz1.sum(axis=0)

                    gnorm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
                    gscale = (5.0 / gnorm) if gnorm > 5.0 else 1.0

                    self.step_t += 1
                    bc1 = 1.0 - 0.9 ** self.step_t
                    bc2 = 1.0 - 0.999 ** self.step_t
                    for k, g in grads.items():
                        if gscale < 1.0:
                            g = g * gscale
                        if k.startswith("W"):
                            self.p[k] *= (1.0 - lr_ep * self.wd)
                        self.m[k] = 0.9 * self.m[k] + 0.1 * g
                        self.v[k] = 0.999 * self.v[k] + 0.001 * (g * g)
                        mh = self.m[k] / bc1
                        vh = self.v[k] / bc2
                        self.p[k] -= lr_ep * mh / (np.sqrt(vh) + 1e-8)
                    if ema_p is not None:
                        d_e = min(self.ema_decay, (1.0 + self.step_t) / (10.0 + self.step_t))
                        for k in ema_p:
                            ema_p[k] = d_e * ema_p[k] + (1.0 - d_e) * self.p[k]

                eval_p = ema_p if ema_p is not None else self.p
                orig_p = self.p
                self.p = eval_p
                va_out, _ = self._forward(Xva, train=False)
                self.p = orig_p
                vloss, _ = self._loss_grad(va_out, yva, wva)
                self.n_iter_ = ep + 1
                if np.isfinite(vloss):
                    if self.topk_avg > 1:
                        if len(top_ckpts) < self.topk_avg or vloss < top_ckpts[-1][0]:
                            top_ckpts.append((vloss, {k: v.copy() for k, v in eval_p.items()}))
                            top_ckpts.sort(key=lambda t: t[0])
                            top_ckpts = top_ckpts[:self.topk_avg]
                    if vloss < best_val - 1e-5:
                        best_val = vloss
                        if self.topk_avg <= 1:
                            top_ckpts = [(vloss, {k: v.copy() for k, v in eval_p.items()})]
                        no_imp = 0
                    else:
                        no_imp += 1
                        if no_imp >= patience:
                            break
            if top_ckpts:
                if len(top_ckpts) == 1:
                    self.p = top_ckpts[0][1]
                else:
                    avg_p = {k: sum(c[1][k] for c in top_ckpts) / len(top_ckpts) for k in self.p}
                    self.p = avg_p
                    vo_avg, _ = self._forward(Xva, train=False)
                    vl_avg, _ = self._loss_grad(vo_avg, yva, wva)
                    if not (np.isfinite(vl_avg) and vl_avg <= top_ckpts[0][0]):
                        self.p = top_ckpts[0][1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self._forward(X, train=False)
        if self.loss_kind in ("huber", "l1", "mse"):
            return out[:, 0].astype(np.float64)
        if self.loss_kind == "bce":
            p1 = 1.0 / (1.0 + np.exp(-np.clip(out[:, 0], -20.0, 20.0)))
            return np.column_stack([1.0 - p1, p1]).astype(np.float64)
        z = out - np.max(out, axis=1, keepdims=True)
        ez = np.exp(np.clip(z, -30.0, 0.0))
        return (ez / np.sum(ez, axis=1, keepdims=True)).astype(np.float64)


def fit_mlp(X, y, Xt, plan: FoldPlan, spec, cfg: ModelConfig, n_jobs: int, time_limit: float,
            seed: int = 0, name: str = "mlp", top_cols: list | None = None,
            max_cols: int = 80, prep_mode: str = "qt", loss_override: str | None = None) -> FitResult | None:
    """Табличная нейросеть ResNet/MLP (обучается строго с нуля на CPU): гауссизация числовых признаков
    (QuantileTransformer или Dual [Quantile+Standard]) + OHE категорий, bias-corrected EMA + Top-3 SWA,
    нативная поддержка L1 (под MAE), Huber, MSE, BCE и CrossEntropy, масштабирование до 600k строк."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

    t0 = time.time()
    guard = _TimeGuard(time_limit)
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
    if not cols or n < 100 or n > 600_000:
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
    tvals = X[plan.time_col].values if plan.time_col and plan.time_col in X.columns else None
    w = sample_weights(y, spec.task_type, cfg, tvals, plan.row_weight).astype(np.float32)
    y_lo = float(np.nanmin(y)) if spec.task_type == "regression" else None
    y_hi = float(np.nanmax(y)) if spec.task_type == "regression" else None
    if spec.task_type == "regression":
        loss_kind = loss_override or ("l1" if spec.metric in ("mae", "medae", "smape") else "mse")
        d_out = 1
    elif spec.task_type == "binary":
        loss_kind, d_out = "bce", 1
    else:
        loss_kind, d_out = "ce", n_classes

    max_fit_rows = 80_000 if n > 120_000 else 120_000

    def _cap_rows(tr_idx, ep_seed: int):
        if len(tr_idx) <= max_fit_rows:
            return tr_idx
        rng_s = np.random.default_rng(ep_seed)
        if spec.task_type != "regression":
            # Stratified subsample preserves rare classes & minority positives on large imbalanced datasets
            yv = yt[tr_idx].astype(int)
            parts = []
            for cls in np.unique(yv):
                ci = tr_idx[yv == cls]
                k_take = max(1, int(round(max_fit_rows * len(ci) / len(tr_idx))))
                parts.append(rng_s.choice(ci, size=min(len(ci), k_take), replace=False) if len(ci) > k_take else ci)
            sel = np.concatenate(parts)
            sel.sort()
            return sel
        if tvals is not None:
            # Prefer recent chronological slice on huge temporal regression tables
            ord_t = np.argsort(np.nan_to_num(tvals[tr_idx], nan=-1e18), kind="mergesort")
            sel = tr_idx[ord_t[-max_fit_rows:]]
            sel.sort()
            return sel
        sel = rng_s.choice(tr_idx, size=max_fit_rows, replace=False)
        sel.sort()
        return sel

    def build_pre(n_fit: int):
        tf = []
        nq = max(10, min(500, n_fit // 2 if n_fit < 500 else max(50, n_fit // 10)))
        if num_cols:
            tf.append(("num", make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                QuantileTransformer(n_quantiles=nq, subsample=50_000, output_distribution="normal", random_state=seed),
            ), num_cols))
            if prep_mode == "dual":
                tf.append(("num_std", make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ), num_cols))
        if cat_cols:
            tf.append(("cat", OneHotEncoder(
                handle_unknown="ignore", min_frequency=10, max_categories=40, sparse_output=False,
            ), cat_cols))
        return ColumnTransformer(tf, sparse_threshold=0.0)

    def fit_one(df_tr, y_tr, w_tr, ep_seed: int):
        pre = build_pre(len(df_tr))
        Z = np.asarray(pre.fit_transform(df_tr), dtype=np.float32)
        Z = np.where(np.isfinite(Z), Z, 0.0)
        if prep_mode == "dual":
            Z = np.clip(Z, -6.0, 6.0)
        if spec.task_type == "regression":
            y_med = float(np.median(y_tr))
            y_sc = float(np.std(y_tr))
            if not np.isfinite(y_sc) or y_sc < 1e-6:
                y_sc = 1.0
            y_norm = ((y_tr - y_med) / y_sc).astype(np.float32)
        else:
            y_med, y_sc = 0.0, 1.0
            y_norm = y_tr.astype(np.float32) if spec.task_type == "binary" else y_tr.astype(np.int64)
        bs = 512 if len(df_tr) > 50_000 else int(min(256, max(32, len(df_tr) // 30)))
        max_ep = 25 if len(df_tr) > 50_000 else (45 if len(df_tr) > 30_000 else 65)
        pat = 8 if len(df_tr) > 50_000 else 12
        net = _NumpyTabularResNet(Z.shape[1], d_out, loss_kind=loss_kind, seed=ep_seed)
        net.fit(Z, y_norm, w_tr, max_epochs=max_ep, batch_size=bs, patience=pat, guard=guard, n_jobs=n_jobs)
        return pre, net, y_med, y_sc

    def pred_one(bundle, df):
        pre, net, y_med, y_sc = bundle
        Z = np.asarray(pre.transform(df), dtype=np.float32)
        Z = np.where(np.isfinite(Z), Z, 0.0)
        if prep_mode == "dual":
            Z = np.clip(Z, -6.0, 6.0)
        p = net.predict(Z)
        if spec.task_type == "regression":
            return np.clip(inverse_target(p * y_sc + y_med, cfg.target_transform), y_lo, y_hi)
        return p

    shape = (n, n_classes) if spec.task_type != "regression" else (n,)
    oof = np.full(shape, np.nan)
    test, iters = [], []
    for k, (tr, va) in enumerate(plan.folds):
        if spec.task_type != "regression" and len(np.unique(yt[tr])) < 2:
            return None
        tr_use = _cap_rows(tr, seed + 17 * k)
        bundle = fit_one(X_in.iloc[tr_use], yt[tr_use], w[tr_use], seed + 17 * k)
        oof[va] = pred_one(bundle, X_in.iloc[va])
        iters.append(int(bundle[1].n_iter_))
        if plan.mode == "cv" and Xt_in is not None:
            test.append(pred_one(bundle, Xt_in))
        if k == 0 and _too_slow(t0, plan, time_limit, name):
            return None
        if guard.expired() and k < len(plan.folds) - 1:
            log(f"{name}: время вышло на фолде {k + 1}/{len(plan.folds)}")
            return _finish(name, oof, None, plan, spec, y, iters, [], t0, cfg, cut=guard.cut)
    if plan.mode == "refit" and Xt_in is not None:
        all_idx = _cap_rows(np.arange(n), seed + 99)
        t_ref0 = time.time()
        b1 = fit_one(X_in.iloc[all_idx], yt[all_idx], w[all_idx], seed + 99)
        p1 = pred_one(b1, Xt_in)
        # On temporal refit, average a second complementary 90/10 bag if plenty of time remains (covers 100% of train)
        dt_ref = time.time() - t_ref0
        if not guard.expired() and (guard.deadline - time.time()) > max(12.0, 1.5 * dt_ref):
            b2 = fit_one(X_in.iloc[all_idx], yt[all_idx], w[all_idx], seed + 199)
            p2 = pred_one(b2, Xt_in)
            test = 0.5 * (p1 + p2)
        else:
            test = p1
    return _finish(name, oof, test, plan, spec, y, iters, [], t0, cfg, cut=guard.cut)


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
