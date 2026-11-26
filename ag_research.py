"""Цикл AutoResearch: гипотеза -> эксперимент -> оценка на собственной валидации -> принять/отклонить.

Гипотезы приходят из двух источников: встроенный список (работает и без LLM) и LLM, которая видит
описание задачи, сводку данных, анализ ошибок и журнал прошлых экспериментов. Любая гипотеза
проверяется одинаково — быстрой LightGBM на фиксированных фолдах, — и принимается только если
улучшает метрику больше порога шума и на большинстве фолдов.
"""

import copy
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ag_features import (entity_key_candidates, entity_lag_cols, entity_periods, op_dup_count, op_entity_history,
                         op_entity_lags, op_freq, op_group_dev, op_nan_count, op_pairwise, op_target_oof, op_text_oof)
from ag_metrics import PROBA_METRICS, TUNABLE, direction, fold_scores, gain, optimize_decision, score
from ag_llm import extract_code_blocks, extract_json
from ag_models import ModelConfig, fit_lgb, make_plan, subset_plan
from ag_utils import elapsed, log, log_exc, python_exe, save_json, short

LGB_PARAM_RANGES = {
    "num_leaves": (4, 512, int),
    "min_child_samples": (1, 2000, int),
    "feature_fraction": (0.1, 1.0, float),
    "bagging_fraction": (0.3, 1.0, float),
    "lambda_l1": (0.0, 100.0, float),
    "lambda_l2": (0.0, 100.0, float),
    "max_depth": (-1, 64, int),
    "min_split_gain": (0.0, 10.0, float),
    "max_bin": (15, 1023, int),
    "min_sum_hessian_in_leaf": (0.0, 100.0, float),
    "path_smooth": (0.0, 100.0, float),
    "cat_smooth": (0.0, 200.0, float),
    "cat_l2": (0.0, 200.0, float),
    "max_cat_to_onehot": (1, 64, int),
    "min_data_per_group": (1, 1000, int),
    "extra_trees": (0, 1, bool),
}
ALIASES = {"min_data_in_leaf": "min_child_samples", "colsample_bytree": "feature_fraction", "subsample":
           "bagging_fraction", "reg_alpha": "lambda_l1", "reg_lambda": "lambda_l2", "min_gain_to_split":
           "min_split_gain", "min_child_weight": "min_sum_hessian_in_leaf"}


def clean_params(params: dict) -> dict:
    out = {}
    for k, v in (params or {}).items():
        k = ALIASES.get(str(k), str(k))
        if k not in LGB_PARAM_RANGES:
            continue
        lo, hi, tp = LGB_PARAM_RANGES[k]
        try:
            v = tp(v) if tp is not bool else bool(v)
        except Exception:
            continue
        if tp is not bool:
            v = min(max(v, lo), hi)
        out[k] = v
    return out


def transform_target_safe(y, how):
    return np.log1p(np.clip(y, 0, None)) if how == "log1p" else y


class Researcher:
    def __init__(self, spec, prep, plan, llm, budget, n_jobs, work: Path, llm_hints=None):
        self.spec, self.prep, self.plan, self.llm = spec, prep, plan, llm
        self.budget, self.n_jobs, self.work = budget, n_jobs, work
        self.llm_hints = llm_hints or {}
        self.y = prep.y
        self.blocks = {}
        self.cfg = ModelConfig(
            target_transform="log1p" if (spec.task_type == "regression" and spec.metric == "rmsle"
                                         and len(prep.y) and float(np.nanmin(prep.y)) >= 0) else "none"
        )
        self.journal = []
        self.best = None
        self.best_score = None
        self.queue = []
        self.tried = set()
        self.eval_times = []
        self.n_rows = len(prep.X)
        self.eval_lr = 0.1 if self.n_rows > 2000 else 0.05
        self.llm_rounds = 0
        self.max_llm_rounds = 5
        self.since_llm = 0
        self.block_cache = {}
        self.custom_counter = 0
        self.n_seeds = 1
        self.empty_llm_rounds = 0
        self.search_tries = 0
        self.search_fails = 0
        self.orig_name = {v: k for k, v in prep.info.get("rename", {}).items()}
        self.shift_note = ""
        # сдвиг долей классов в тесте (LabelShift): если он учтён, гипотезы оцениваются под распределение теста
        self.label_shift = None
        self.best_seed1 = None  # скор текущей лучшей конфигурации со вторым сидом (для перепроверки)
        self.alt_plans = {}
        self.retry = []
        self.has_lags = False
        if spec.task_type == "regression" and spec.metric in ("rmsle",) and np.nanmin(self.y) >= 0:
            self.cfg.target_transform = "log1p"
        # на больших данных эксперименты идут на подвыборке строк (финальные модели — на всех данных)
        self.sub = None
        self.ey, self.eplan = self.y, plan
        limit_rows = 250_000
        if self.n_rows > limit_rows:
            rng = np.random.default_rng(0)
            self.sub = np.sort(rng.choice(self.n_rows, limit_rows, replace=False))
            self.ey = self.y[self.sub]
            self.eplan = subset_plan(plan, self.sub)
            log(f"исследование на подвыборке {limit_rows} из {self.n_rows} строк")

    # ------------------------------------------------------------ данные и оценка

    def frames(self, extra=None, blocks=None):
        blocks = self.blocks if blocks is None else blocks
        parts_a = [self.prep.X] + [a for a, _ in blocks.values()]
        parts_b = [self.prep.Xt] + [b for _, b in blocks.values()]
        if extra is not None:
            parts_a.append(extra[0])
            parts_b.append(extra[1])
        X = pd.concat([p.reset_index(drop=True) for p in parts_a], axis=1)
        Xt = pd.concat([p.reset_index(drop=True) for p in parts_b], axis=1)
        dup = X.columns.duplicated()
        return X.loc[:, ~dup], Xt.loc[:, ~dup]

    def effective_score(self, r):
        """Скор с подобранным решающим правилом (то, что реально уйдёт в ответ); фолды — с тем же правилом.

        При учтённом сдвиге долей классов — на пересчитанных вероятностях и со взвешенными строками, то есть
        оценка метрики на тесте: гипотеза, полезная на train, может вредить на сдвинутом тесте (на UNSW-NB15
        параметры с +0.001 по OOF стоили −0.01 под долями теста)."""
        mask = self.eplan.oof_mask & ~np.isnan(r.oof).reshape(len(r.oof), -1).any(1)
        if self.spec.task_type != "regression" and self.spec.metric in TUNABLE and self.spec.metric not in PROBA_METRICS:
            oof, sw = r.oof, None
            ls = self.label_shift
            if ls is not None and ls.active and self.sub is None and len(ls.y) == len(r.oof):
                oof, _, sw = ls.apply(r.oof)
            d = optimize_decision(self.spec.metric, self.ey[mask], oof[mask], rounds=1,
                                  sample_weight=None if sw is None else sw[mask])
            r.fold_scores = fold_scores(self.spec.metric, self.spec.task_type, self.ey, oof, self.eplan.folds, d, sw)
            return float(d.get("score", r.score))
        return float(r.score)

    def evaluate(self, X, cfg, label: str, time_limit=None, Xt=None, seed: int = 0):
        limit = time_limit or max(60.0, min(900.0, self.budget.left() * 0.25))
        t0 = time.time()
        if self.sub is not None:
            X = X.iloc[self.sub].reset_index(drop=True)
        r = fit_lgb(X, self.ey, Xt, self.eplan, self.spec, cfg, self.n_jobs, limit, lr=self.eval_lr, es=50,
                    predict_test=Xt is not None, seed=seed, name=label)
        # на быстрых данных усредняем несколько сидов: разница между гипотезами меньше тонет в шуме
        if self.n_seeds > 1 and not np.isnan(r.oof[self.eplan.oof_mask]).any():
            runs = [r]
            for sd in range(1, self.n_seeds):
                rr = fit_lgb(X, self.ey, Xt, self.plan_for(sd), self.spec, cfg, self.n_jobs, limit, lr=self.eval_lr,
                             es=50, predict_test=Xt is not None, seed=seed + sd, name=label)
                if np.isnan(rr.oof[self.eplan.oof_mask]).any():
                    break
                runs.append(rr)
            if len(runs) > 1:
                r.oof = sum(x.oof for x in runs) / len(runs)
                if r.test is not None and all(x.test is not None for x in runs):
                    r.test = sum(x.test for x in runs) / len(runs)
                r.importance = sum(x.importance for x in runs) / len(runs)
                m = self.eplan.oof_mask
                r.score = score(self.spec.metric, self.spec.task_type, self.ey[m], r.oof[m])
                r.fold_scores = fold_scores(self.spec.metric, self.spec.task_type, self.ey, r.oof, self.eplan.folds)
        dt = time.time() - t0
        self.eval_times.append(dt)
        if np.isnan(r.oof[self.eplan.oof_mask]).any():
            return None, r
        return self.effective_score(r), r

    def plan_for(self, sd: int):
        """Разбиение для дополнительного сида: при обычной CV — другое (повторная CV усредняет и шум разбиения на
        фолды, а не только шум модели), при валидации по времени — то же самое."""
        if sd == 0 or self.eplan.mode != "cv":
            return self.eplan
        if sd not in self.alt_plans:
            Xs = self.prep.X if self.sub is None else self.prep.X.iloc[self.sub].reset_index(drop=True)
            self.alt_plans[sd] = make_plan(Xs, self.ey, self.spec.task_type, seed=42 + 101 * sd)
        return self.alt_plans[sd]

    def tolerance(self):
        b = abs(self.best_score or 0.0)
        if direction(self.spec.metric) == "max" and b <= 1.0:
            # чем ближе к идеалу, тем меньше реальные приросты: порог от оставшегося зазора
            tol = min(0.001, max(0.0002, 0.01 * (1.0 - b)))
        else:
            tol = max(1e-6, 0.002 * b)
        # повторная CV (несколько разбиений) шумит примерно в корень из числа разбиений раз меньше
        return tol * 0.6 if (self.n_seeds > 1 and self.eplan.mode == "cv") else tol

    def est_eval_time(self):
        if not self.eval_times:
            return 60.0
        return float(np.median(self.eval_times[-5:])) * 1.3 + 5

    # ------------------------------------------------------------ гипотезы

    def add(self, h, front=False):
        key = json.dumps({k: h.get(k) for k in ("kind", "args")}, sort_keys=True, default=str)
        if key in self.tried:
            return
        self.tried.add(key)
        if front:
            self.queue.insert(0, h)
        else:
            self.queue.append(h)

    def builtin(self):
        spec, X = self.spec, self.prep.X
        if self.prep.raw_text:
            self.add({"kind": "text_oof", "args": {"cols": list(self.prep.raw_text)}, "title":
                      "TF-IDF + линейная модель по текстам (OOF-прогноз как признак)", "source": "builtin"})
        if self.plan.time_col:
            extra = self.llm_hints.get("entity_columns") if isinstance(self.llm_hints.get("entity_columns"), list) else None
            if extra:
                ren = self.prep.info.get("rename", {})
                extra = [ren.get(str(c), str(c)) for c in extra]
            try:
                keys = entity_key_candidates(X, self.plan.time_col, self.prep.cat_cols, extra=extra)
            except Exception:
                log_exc("entity_key_candidates")
                keys = []
            for k in keys:
                self.add({"kind": "entity_history", "args": {"key": k}, "title": f"история таргета объекта по ключу {k}",
                          "source": "builtin"})
        if spec.task_type != "regression" and spec.metric in TUNABLE:
            self.add({"kind": "class_weight", "args": {"power": 0.5}, "title": "веса классов (обратная частота)^0.5",
                      "source": "builtin"})
        if spec.task_type == "regression":
            if spec.metric in ("rmse", "mse", "r2"):
                self.add({"kind": "objective", "args": {"variant": "huber"}, "title": "Huber-потеря против выбросов",
                          "source": "builtin"})
            yy = self.y
            if (np.nanmin(yy) >= 0 and pd.Series(yy).skew() > 1.0 and spec.metric not in ("mae", "medae")
                    and self.cfg.target_transform != "log1p"):
                self.add({"kind": "target_transform", "args": {"how": "log1p"}, "title": "обучение на log1p(таргета)",
                          "source": "builtin"})
        if self.plan.time_col:
            self.add({"kind": "recency_weight", "args": {"strength": 1.0}, "title": "больший вес свежим периодам",
                      "source": "builtin"})
        if self.prep.cat_cols:
            self.add({"kind": "freq", "args": {"cols": list(self.prep.cat_cols)}, "title": "частотное кодирование категорий",
                      "source": "builtin"})
        hc = self.high_card_cols()
        if hc:
            self.add({"kind": "target_oof", "args": {"cols": hc}, "title": "кодирование категорий с сотнями значений "
                      "таргетом по чужим фолдам", "source": "builtin"})
        vf = self.value_freq_cols()
        if vf:
            # частота точного значения: типовые значения (стандартные размеры пакетов, площади) повторяются,
            # случайные — нет; на UNSW-NB15 это разделяет «нормальный» трафик и фаззинг (+0.01 macro-F1)
            self.add({"kind": "freq", "args": {"cols": vf}, "title": "частота точного значения числовых признаков (train+test)",
                      "source": "builtin"})
        if X.isna().any().any():
            self.add({"kind": "nan_count", "args": {}, "title": "число пропусков в строке", "source": "builtin"})
        try:
            sample = X.sample(min(len(X), 50000), random_state=0)
            if sample.duplicated().mean() > 0.01:
                self.add({"kind": "dup_count", "args": {}, "title": "число полных дублей строки (train+test)",
                          "source": "builtin"})
        except Exception:
            pass
        self.add({"kind": "group_dev", "args": {}, "title": "отклонение числовых признаков от среднего по категории",
                  "source": "builtin"})
        self.add({"kind": "pairwise", "args": {}, "title": "отношения и разности пар важных числовых признаков",
                  "source": "builtin"})
        self.add({"kind": "params", "args": {"params": {"num_leaves": 127, "min_child_samples": 20}},
                  "title": "больше ёмкости деревьев", "source": "builtin"})
        self.add({"kind": "params", "args": {"params": {"num_leaves": 15, "min_child_samples": 100, "lambda_l2": 10.0}},
                  "title": "сильнее регуляризация", "source": "builtin"})
        self.add({"kind": "params", "args": {"params": {"feature_fraction": 0.4}}, "title": "меньше признаков на дерево",
                  "source": "builtin"})
        self.add({"kind": "drop_low_importance", "args": {}, "title": "убрать бесполезные признаки", "source": "builtin"})
        if spec.task_type != "regression" and spec.metric in TUNABLE:
            self.add({"kind": "class_weight", "args": {"power": 1.0}, "title": "веса классов (обратная частота)^1",
                      "source": "builtin"})

    def materialize(self, h):
        """Гипотеза -> (новый блок признаков или None, новая конфигурация или None)."""
        kind, args = h["kind"], h.get("args", {})
        cfg = copy.deepcopy(self.cfg)
        X, Xt = self.prep.X, self.prep.Xt
        if kind == "entity_history":
            key = [c for c in args.get("key", []) if c in X.columns and c != self.plan.time_col]
            if not key or not self.plan.time_col:
                return None, None
            t_tr, t_te = entity_periods(X, Xt, self.plan.time_col)
            base = op_entity_history(X, Xt, self.y, key, self.plan.time_col, t_tr, t_te)
            if self.spec.task_type != "multiclass":
                cols = [] if self.has_lags else entity_lag_cols(X, self.y, key, t_tr, exclude=[self.plan.time_col])
                if cols:
                    # и прошлые значения признаков объекта: изменение признака (рейтинга, нагрузки) объясняет
                    # изменение таргета. Отдельной гипотезой не проверить: на WiDS выигрыш −0.48 RMSE смеси по трём
                    # сидам, но один сид быстрой модели без поправки к прошлому значению его не видит
                    la, lb = op_entity_lags(X, Xt, key, cols, t_tr, t_te)
                    base = (pd.concat([base[0], la], axis=1), pd.concat([base[1], lb], axis=1))
                return base, None
            # мультикласс: среднее номеров классов бессмысленно — берём долю каждого класса в истории объекта
            keep = ["hist__cnt", "hist__gap"]
            parts_a, parts_b = [base[0][keep]], [base[1][keep]]
            for k in range(min(len(self.spec.classes), 15)):
                a, b = op_entity_history(X, Xt, (self.y == k).astype(float), key, self.plan.time_col, t_tr, t_te)
                parts_a.append(a[["hist__mean", "hist__last"]].add_suffix(f"_c{k}"))
                parts_b.append(b[["hist__mean", "hist__last"]].add_suffix(f"_c{k}"))
            return (pd.concat(parts_a, axis=1), pd.concat(parts_b, axis=1)), None
        if kind == "text_oof":
            raw = {c: v for c, v in self.prep.raw_text.items() if c in args.get("cols", [])}
            if not raw:
                return None, None
            n_classes = len(self.spec.classes) if self.spec.task_type != "regression" else 1
            yy = self.y if self.spec.task_type != "regression" else transform_target_safe(self.y, self.cfg.target_transform)
            periods = None
            if self.plan.time_col:
                periods, _ = entity_periods(X, Xt, self.plan.time_col)
            return op_text_oof(raw, yy, self.spec.task_type, n_classes, self.n_jobs, periods=periods), None
        if kind == "freq":
            cols = [c for c in args.get("cols", []) if c in X.columns]
            return (op_freq(X, Xt, cols), None) if cols else (None, None)
        if kind == "target_oof":
            cols = [c for c in args.get("cols", []) if c in X.columns]
            if not cols or self.plan.time_col:
                return None, None
            n_classes = len(self.spec.classes) if self.spec.task_type != "regression" else 1
            return op_target_oof(X, Xt, self.y, cols, self.plan.folds, n_classes), None
        if kind == "nan_count":
            return op_nan_count(X, Xt), None
        if kind == "dup_count":
            return op_dup_count(X, Xt), None
        if kind == "group_dev":
            pairs = args.get("pairs") or self.auto_pairs()
            pairs = [(a, b) for a, b in pairs if a in X.columns and b in X.columns and a != b
                     and pd.api.types.is_numeric_dtype(X[b])]
            return (op_group_dev(X, Xt, pairs[:12]), None) if pairs else (None, None)
        if kind == "pairwise":
            cols = args.get("cols") or self.top_numeric(5)
            cols = [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
            return (op_pairwise(X, Xt, cols[:6]), None) if len(cols) >= 2 else (None, None)
        if kind == "custom":
            return self.run_custom(args.get("code", ""), h), None
        if kind == "params":
            p = clean_params(args.get("params", {}))
            if not p:
                return None, None
            cfg.lgb_params = {**cfg.lgb_params, **p}
            return None, cfg
        if kind == "objective" and self.spec.task_type == "regression":
            v = args.get("variant")
            cfg.objective_variant = v if v in ("huber", "fair") else None
            if args.get("alpha"):
                try:
                    cfg.huber_alpha = float(args["alpha"])
                except Exception:
                    pass
            return None, cfg
        if kind == "target_transform" and self.spec.task_type == "regression":
            how = args.get("how", "none")
            if how == "log1p" and np.nanmin(self.y) < 0:
                return None, None
            cfg.target_transform = how if how in ("none", "log1p") else "none"
            return None, cfg
        if kind == "class_weight" and self.spec.task_type != "regression":
            cfg.class_weight_power = float(min(max(float(args.get("power", 0.5)), 0.0), 1.5))
            return None, cfg
        if kind == "recency_weight" and self.plan.time_col:
            cfg.recency_weight = float(min(max(float(args.get("strength", 1.0)), 0.0), 5.0))
            return None, cfg
        if kind == "drop_features":
            cols = [c for c in args.get("columns", []) if c in X.columns or any(c in a.columns for a, _ in self.blocks.values())]
            if not cols:
                return None, None
            cfg.drop = sorted(set(cfg.drop) | set(cols))
            return None, cfg
        if kind == "drop_low_importance":
            if self.best is None or self.best.importance is None:
                return None, None
            imp = self.best.importance
            low = imp[imp <= imp.sum() * 0.0005].index.tolist()
            if len(low) < 2 or len(low) > 0.7 * len(imp):
                return None, None
            cfg.drop = sorted(set(cfg.drop) | set(low))
            return None, cfg
        return None, None

    def high_card_cols(self, min_unique: int = 100, max_cols: int = 10):
        """Категории с сотнями значений (и коды очень частых категорий) — для кодирования таргетом.
        При валидации по времени не нужно: там тот же смысл несёт история группы по прошлым периодам."""
        if self.plan.time_col or (self.spec.task_type == "multiclass" and len(self.spec.classes) > 12):
            return []
        X = self.prep.X
        cols = [c for c in self.prep.cat_cols if X[c].nunique() >= min_unique]
        cols += [c for c in X.columns if f"{c}__freq" in X.columns and c not in cols]
        return cols[:max_cols]

    def value_freq_cols(self, max_cols: int = 40):
        """Числовые колонки с повторяющимися значениями: не почти константы и не почти все уникальные."""
        X, Xt = self.prep.X, self.prep.Xt
        n = len(X) + len(Xt)
        out = []
        for c in X.columns:
            if c == self.plan.time_col or c in self.prep.cat_cols or not pd.api.types.is_numeric_dtype(X[c]):
                continue
            nu = pd.concat([X[c], Xt[c]], ignore_index=True).nunique()
            if 20 <= nu <= 0.5 * n:
                out.append(c)
        return out[:max_cols]

    def top_numeric(self, k: int):
        if self.best is None or self.best.importance is None:
            return []
        imp = self.best.importance.sort_values(ascending=False)
        return [c for c in imp.index if c in self.prep.X.columns and c not in self.prep.cat_cols
                and pd.api.types.is_numeric_dtype(self.prep.X[c]) and c != self.plan.time_col
                and self.prep.X[c].nunique() > 10][:k]

    def auto_pairs(self):
        if self.best is None or self.best.importance is None:
            return []
        imp = self.best.importance.sort_values(ascending=False)
        cats = [c for c in imp.index if c in self.prep.cat_cols][:3]
        nums = [c for c in imp.index if c not in self.prep.cat_cols and c in self.prep.X.columns
                and pd.api.types.is_numeric_dtype(self.prep.X[c]) and c != self.plan.time_col][:4]
        return [(a, b) for a in cats for b in nums]

    def run_custom(self, code: str, h):
        if not code or "make_features" not in code:
            return None
        out, crash = self._run_worker(code, h)
        if crash and self.llm is not None and self.llm.enabled:
            # код от LLM часто падает на одной строке и теряет весь блок: одна попытка исправить по трейсбеку
            fixed = self.repair_code(code, crash)
            if fixed:
                out, _ = self._run_worker(fixed, h)
                if out is not None:
                    h["repaired"] = True
                    h.pop("error", None)
                    h.setdefault("args", {})["code"] = fixed  # повтор почти принятой гипотезы — уже без ошибки
                    log("код признаков исправлен по трейсбеку")
        return out

    def repair_code(self, code: str, error: str):
        dtypes = ", ".join(f"{c}:{'category' if c in self.prep.cat_cols else t}" for c, t in self.prep.X.dtypes.items())
        sys_prompt = ("You fix pandas feature-engineering code. Reply with only one ```python fenced block containing the "
                      "full corrected function make_features(train, test).")
        user = f"""This function raised an error:
```python
{code}
```
Error (end of traceback):
{short(error, 1200)}

Columns of train/test with dtypes: {short(dtypes, 4000)}
make_features must return (new_train_df, new_test_df) with ONLY new numeric columns, same row order. pandas as pd and
numpy as np are available. Fix this error and any similar problems; keep all other features."""
        text = self.llm.chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                             max_tokens=8000, temperature=0.2)
        blocks = [b for b in extract_code_blocks(text) if "make_features" in b]
        return blocks[0] if blocks else None

    def _run_worker(self, code: str, h):
        """Код признаков в отдельном процессе -> ((train, test) или None, трейсбек, если код упал с ошибкой)."""
        self.custom_counter += 1
        d = self.work / f"custom_{self.custom_counter}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "code.py").write_text(code, encoding="utf-8")
        base_a = self.work / "base_train.pkl"
        base_b = self.work / "base_test.pkl"
        if not base_a.exists():
            self.prep.X.to_pickle(base_a)
            self.prep.Xt.to_pickle(base_b)
        for src, dst in ((base_a, d / "train.pkl"), (base_b, d / "test.pkl")):
            try:
                os.link(src, dst)
            except Exception:
                dst.write_bytes(src.read_bytes())
        worker = Path(__file__).resolve().parent / "ag_worker.py"
        env = dict(os.environ)
        env.update({"OMP_NUM_THREADS": str(self.n_jobs), "PYTHONIOENCODING": "utf-8"})
        timeout = max(30.0, min(300.0, self.budget.left() * 0.05))
        try:
            res = subprocess.run([python_exe(), "-u", str(worker), str(d)], capture_output=True, text=True,
                                 timeout=timeout, env=env, cwd=str(d))
        except subprocess.TimeoutExpired:
            h["error"] = f"код признаков не уложился в {timeout:.0f}с"
            log(h["error"])
            return None, None
        finally:
            for p in (d / "train.pkl", d / "test.pkl"):
                try:
                    p.unlink()
                except Exception:
                    pass
        if res.returncode != 0:
            h["error"] = (res.stderr or res.stdout or "").strip()[-700:]
            log("код признаков упал:", h["error"].splitlines()[-1] if h["error"] else "")
            return None, h["error"]
        try:
            a = pd.read_pickle(d / "out_train.pkl")
            b = pd.read_pickle(d / "out_test.pkl")
        except Exception:
            h["error"] = "не удалось прочитать результат кода"
            return None, None
        prefix = f"llm{self.custom_counter}__"
        a.columns = [prefix + str(c) for c in a.columns]
        b.columns = [prefix + str(c) for c in b.columns]
        log("код признаков:", (res.stdout or "").strip()[-300:])
        return (a, b), None

    # ------------------------------------------------------------ эксперимент

    def baseline(self, on_result=None):
        X, Xt = self.frames()
        s, r = self.evaluate(X, self.cfg, "baseline", time_limit=max(120.0, self.budget.left() * 0.2), Xt=Xt)
        if on_result is not None and r.test is not None:
            try:
                on_result(r)
            except Exception:
                log_exc("on_result")
            if s is not None and self.label_shift is not None and self.label_shift.active:
                s = self.effective_score(r)  # дальше все гипотезы сравниваются под доли классов теста
                log(f"исследование оценивает гипотезы под оценённые доли классов теста: база {s:.5f}")
        self.best, self.best_score = r, s
        if s is None:
            # не успели даже один прогон: исследование бессмысленно, всё время — финальной модели
            log("базовый прогон не завершился за отведённое время — исследование пропущено")
            self.journal.append({"id": 0, "title": "базовая LightGBM", "source": "builtin", "score": None,
                                 "accepted": False, "error": "не уложилась во время"})
            return None
        if r.seconds < 8:
            self.n_seeds = 3
            X, _ = self.frames()
            s, r = self.evaluate(X, self.cfg, "baseline3")
            self.best, self.best_score = r, s
            log("эксперименты быстрые: каждая гипотеза оценивается по среднему трёх сидов")
        self.journal.append({"id": 0, "title": "базовая LightGBM", "source": "builtin", "score": s, "raw": r.score,
                             "folds": r.fold_scores, "accepted": True, "seconds": round(r.seconds, 1)})
        log(f"БАЗА: {self.spec.metric}={s:.5f} (сырой {r.score:.5f}) фолды={np.round(r.fold_scores, 5).tolist()} "
            f"итерации={r.iters} за {r.seconds:.1f}s")
        return s

    def try_one(self, h):
        hid = len(self.journal)
        entry = {"id": hid, "title": h.get("title", h["kind"]), "source": h.get("source"), "kind": h["kind"],
                 "args": {k: (v if k != "code" else short(v, 300)) for k, v in h.get("args", {}).items()},
                 "rationale": h.get("rationale")}
        t0 = time.time()
        try:
            block, cfg = self.materialize(h)
        except Exception as e:
            log_exc("materialize")
            block, cfg = None, None
            h["error"] = repr(e)[:400]
        if h.get("repaired"):
            entry["repaired"] = True
        if block is None and cfg is None:
            entry.update(accepted=False, error=h.get("error", "гипотеза неприменима"), seconds=round(time.time() - t0, 1))
            self.journal.append(entry)
            err_line = str(entry["error"]).strip().splitlines()[-1] if entry["error"] else ""
            log(f"#{hid} [{h.get('source')}] {entry['title']}: пропуск ({short(err_line, 200)})")
            return False
        if block is not None:
            # одинаковые имена у разных гипотез (например, история по разным ключам) не должны склеиваться
            existing = set(self.frames()[0].columns)
            ren = {c: f"{c}__h{hid}" for c in block[0].columns if c in existing}
            if ren:
                block = (block[0].rename(columns=ren), block[1].rename(columns=ren))
        X, _ = self.frames(extra=block)
        cfg = cfg or self.cfg
        s, r = self.evaluate(X, cfg, f"h{hid}")
        if s is None:
            entry.update(accepted=False, error="не уложилось во время", seconds=round(time.time() - t0, 1))
            self.journal.append(entry)
            return False
        d = gain(s, self.best_score, self.spec.metric)
        wins = sum(gain(a, b, self.spec.metric) > 0 for a, b in zip(r.fold_scores, self.best.fold_scores))
        n_f = len(r.fold_scores)
        tol = self.tolerance()
        # источники с множеством попыток (LLM, случайный поиск, повтор почти принятых): чем больше попыток,
        # тем вероятнее принять шум валидации за улучшение — для них правило строже
        strict = h.get("source") in ("llm", "search") or bool(h.get("retried"))
        if n_f <= 2:
            # два фолда (валидация по времени): нужны оба, либо явный выигрыш на последнем периоде
            last_better = n_f > 0 and gain(r.fold_scores[0], self.best.fold_scores[0], self.spec.metric) > 0
            accepted = d > tol and (wins == n_f or (not strict and last_better and d > 3 * tol))
        else:
            accepted = d > tol and wins >= math.ceil(n_f / 2) + (1 if strict else 0)
        s_seed1 = None
        if (accepted and d < (5 if strict else 3) * tol and self.n_seeds == 1
                and self.budget.left() > 6 * self.est_eval_time()):
            # пограничный выигрыш: шум одного сида сравним с порогом — перепроверка вторым сидом
            s_seed1, _ = self.evaluate(X, cfg, f"h{hid}s1", seed=1)
            if self.best_seed1 is None:
                Xb, _ = self.frames()
                self.best_seed1, _ = self.evaluate(Xb, self.cfg, "best_s1", seed=1)
            d2 = gain(s_seed1, self.best_seed1, self.spec.metric) if (
                s_seed1 is not None and self.best_seed1 is not None) else float("-inf")
            accepted = (d + d2) / 2 > tol
            entry["recheck_delta"] = d2
            log(f"#{hid}: пограничный выигрыш {d:+.5f}, со вторым сидом {d2:+.5f} -> "
                f"{'подтверждён' if accepted else 'не подтвердился'}")
        entry.update(score=s, raw=r.score, delta=d, folds=r.fold_scores, fold_wins=f"{wins}/{len(r.fold_scores)}",
                     accepted=bool(accepted), seconds=round(time.time() - t0, 1),
                     new_features=list(block[0].columns)[:30] if block is not None else [])
        self.journal.append(entry)
        log(f"#{hid} [{h.get('source')}] {entry['title']}: {self.spec.metric}={s:.5f} "
            f"Δ={d:+.5f} фолды {wins}/{len(r.fold_scores)} -> {'ПРИНЯТО' if accepted else 'отклонено'}")
        # другой ключ объекта перепроверяется, даже если сначала проиграл: его польза сильно зависит от потерь и
        # признаков, найденных позже (WiDS: ключ с типом здания в начале −0.50, а в финальных моделях смесь −0.08)
        retry_ok = d > 0 or (h["kind"] == "entity_history" and d > -0.02 * abs(self.best_score or 0.0))
        if not accepted and retry_ok and block is not None and not h.get("retried"):
            # почти прошла: перепроверим в конце, когда конфигурация модели уже изменится
            self.retry.append({**h, "retried": True, "title": h.get("title", h["kind"]) + " (повтор)"})
        if accepted:
            if block is not None:
                self.blocks[f"h{hid}"] = block
            self.cfg = cfg
            self.best, self.best_score = r, s
            self.best_seed1 = s_seed1
            if block is not None and any(c.endswith("__dlag") for c in block[0].columns):
                self.has_lags = True  # прошлые признаки — один раз, по первому принятому ключу объекта
        return accepted

    # ------------------------------------------------------------ LLM

    def context_summary(self, max_cols: int = 60):
        X, y, spec = self.prep.X, self.y, self.spec
        imp = self.best.importance.sort_values(ascending=False) if self.best is not None and self.best.importance is not None else None
        rows = []
        cols = list(imp.index[:max_cols]) if imp is not None else list(X.columns[:max_cols])
        for c in cols:
            if c not in X.columns:
                continue
            s = X[c]
            kind = "cat" if c in self.prep.cat_cols else "num"
            info = f"{c} [{kind}] nunique={s.nunique()} miss={s.isna().mean():.2f}"
            if kind == "num":
                info += f" min={s.min():.4g} median={s.median():.4g} max={s.max():.4g}"
            else:
                info += f" top={list(s.value_counts().index[:3])}"
            if imp is not None:
                info += f" gain_share={imp.get(c, 0) / max(imp.sum(), 1e-9):.3f}"
            desc = spec.descriptions.get(c) or spec.descriptions.get(self.orig_name.get(c, c))
            if desc:
                info += f" — {short(desc, 90)}"
            rows.append(info)
        if spec.task_type == "regression":
            tgt = f"target {spec.target}: mean={np.mean(y):.4g} std={np.std(y):.4g} min={np.min(y):.4g} max={np.max(y):.4g} skew={pd.Series(y).skew():.2f}"
        else:
            cnt = np.bincount(y)
            tgt = "classes: " + ", ".join(f"{spec.classes[i]}={cnt[i]}" for i in range(len(cnt)))
        return tgt, rows

    def error_analysis(self):
        r = self.best
        if r is None:
            return ""
        mask = self.eplan.oof_mask & ~np.isnan(r.oof).reshape(len(r.oof), -1).any(1)
        y = self.ey[mask]
        Xe = self.prep.X if self.sub is None else self.prep.X.iloc[self.sub].reset_index(drop=True)
        X = Xe[mask]
        out = []
        try:
            if self.spec.task_type == "regression":
                err = r.oof[mask] - y
                out.append(f"mean residual={err.mean():.4g}, |err| p50={np.median(np.abs(err)):.4g} p95={np.quantile(np.abs(err), 0.95):.4g}")
                big = np.abs(err) > np.quantile(np.abs(err), 0.95)
                out.append(f"top-5% errors: target mean {y[big].mean():.4g} vs overall {y.mean():.4g}")
                for c in self.prep.cat_cols[:3]:
                    g = pd.DataFrame({"c": X[c].astype(str).values, "e2": err ** 2}).groupby("c").e2.agg(["mean", "size"])
                    g = g[g["size"] >= 30].sort_values("mean", ascending=False).head(4)
                    out.append(f"worst {c}: " + ", ".join(f"{i} (rmse={math.sqrt(v):.3g}, n={n})" for i, (v, n) in g.iterrows()))
            else:
                from sklearn.metrics import confusion_matrix, f1_score

                oof, sw = r.oof, None
                ls = self.label_shift
                if ls is not None and ls.active and self.sub is None and len(ls.y) == len(r.oof):
                    # ошибки — под оценённые доли классов теста: на сдвинутом тесте важны другие пары классов
                    oof, _, sw_all = ls.apply(r.oof)
                    sw = sw_all[mask]
                    est = ls.info.get("test_counts_est", [])
                    out.append("estimated test class counts (train counts): " + ", ".join(
                        f"{c}={e} ({n})" for c, e, n in zip(self.spec.classes, est, np.bincount(y, minlength=len(est)))))
                pred = oof[mask].argmax(1)
                f1 = f1_score(y, pred, average=None, labels=np.arange(len(self.spec.classes)), zero_division=0,
                              sample_weight=sw)
                out.append("per-class F1: " + ", ".join(f"{self.spec.classes[i]}={f1[i]:.3f}" for i in range(len(f1))))
                cm = confusion_matrix(y, pred, labels=np.arange(len(self.spec.classes)), sample_weight=sw)
                cm = np.round(cm).astype(int)
                np.fill_diagonal(cm, 0)
                flat = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0][:5]
                out.append("top confusions (true->pred): " + ", ".join(
                    f"{self.spec.classes[a]}->{self.spec.classes[b]}={cm[a, b]}" for a, b in flat if cm[a, b] > 0))
        except Exception:
            log_exc("error_analysis")
        return "\n".join(out)

    def llm_round(self, n: int = 4):
        if self.llm is None or not self.llm.enabled:
            return 0
        self.llm_rounds += 1
        self.since_llm = 0
        spec = self.spec
        tgt, rows = self.context_summary()
        hist = []
        for e in [e for e in self.journal if e.get("kind") != "note"][-18:]:
            hist.append(f"#{e['id']} {e['title']} [{e.get('kind', '')}] -> score={e.get('score')} "
                        f"delta={e.get('delta')} accepted={e.get('accepted')}" + (f" error={short(e['error'], 160)}" if e.get("error") else ""))
        cat_list = list(self.prep.cat_cols)[:30]
        accepted_feats = [c for a, _ in self.blocks.values() for c in a.columns][:40]
        sys_prompt = ("You are an autonomous ML research agent. You improve a tabular model by proposing small, testable "
                      "hypotheses. Each hypothesis is evaluated automatically on fixed validation folds and kept only if "
                      "the metric improves. Think about WHY the model errs. Reply with one JSON object, followed only by the "
                      "python code blocks it references.")
        user = f"""TASK: {spec.task_type}, metric={spec.metric} ({direction(spec.metric)}imize). Target: {spec.target}.
Statement excerpt:
{short(spec.text, 2500)}

Notes extracted earlier: {short(spec.notes, 1400)}
Initial ideas after reading the statement: {short(self.llm_hints.get("hypotheses") or [], 800)}
Rows: train={len(self.prep.X)}, test={len(self.prep.Xt)}. Validation: {self.plan.desc}.
{self.shift_note}
{tgt}
Columns (sorted by importance in the current best model):
{chr(10).join(rows)}
Categorical columns: {cat_list}
Already accepted engineered features: {accepted_feats}
Current config: {json.dumps(self.cfg.to_dict(), default=str)[:600]}
Current best {spec.metric}: {self.best_score}
Error analysis of the current best model:
{self.error_analysis()}

Experiment journal (latest last):
{chr(10).join(hist)}

Propose up to {n} NEW hypotheses that were not tried yet. Allowed action types (use exactly these JSON shapes):
- {{"type":"custom_features","code":"BLOCK_1"}}  the referenced python block must define make_features(train, test) that takes two pandas DataFrames with the feature columns listed above (no target!) and returns (new_train_df, new_test_df) containing ONLY new numeric columns, same row order. Categorical columns have pandas "category" dtype (convert with .astype(str) or .cat.codes before arithmetic). pandas as pd and numpy as np are available; no other imports needed. Row-wise formulas, ratios, differences, aggregates across related columns, group statistics computed on concatenated train+test are good ideas. Keep it vectorized and fast.
- {{"type":"params","params":{{lightgbm params: num_leaves, min_child_samples, feature_fraction, bagging_fraction, lambda_l1, lambda_l2, max_depth, min_split_gain, max_bin, path_smooth, cat_smooth, extra_trees}}}}
- {{"type":"drop_features","columns":[...]}}
- {{"type":"frequency_encoding","columns":[...]}}
- {{"type":"group_deviation","pairs":[["<categorical>","<numeric>"],...]}}
{'- {"type":"objective","objective":"l2|huber|fair"}  and  {"type":"target_transform","transform":"none|log1p"}' if spec.task_type == 'regression' else '- {"type":"class_weight","power":0.0-1.5}'}
{'- {"type":"entity_history","key_columns":[...]}  (target history of the same object in previous periods)' + chr(10) + '- {"type":"recency_weight","strength":0-3}' if self.plan.time_col else ''}
Return: {{"analysis":"<2-4 sentences: what limits the model now>","hypotheses":[{{"title":"<short>","rationale":"<why>","action":{{...}}}}]}}
For custom_features put the code itself in a separate ```python fenced block AFTER the JSON and write "code":"BLOCK_1" (BLOCK_2 for the second such hypothesis, and so on)."""
        text = self.llm.chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                             max_tokens=12000, temperature=0.5)
        if not text:
            return 0
        obj = extract_json(text)
        if not isinstance(obj, dict):
            log("LLM: не удалось разобрать JSON:", short(text, 400))
            return 0
        if not isinstance(obj.get("hypotheses"), list):
            log("LLM: в ответе нет списка hypotheses, ключи:", list(obj)[:10])
        self.code_blocks = extract_code_blocks(text)
        if obj.get("analysis"):
            log("LLM-анализ:", short(obj.get("analysis"), 600))
            self.journal.append({"id": len(self.journal), "title": "LLM analysis", "source": "llm", "kind": "note",
                                 "analysis": short(obj.get("analysis"), 1500), "accepted": None})
        added = 0
        items = [h for h in (self.normalize(it) for it in (obj.get("hypotheses") or [])[:n]) if h is not None]
        for h in reversed(items):  # в начало очереди, сохраняя порядок, предложенный LLM
            before = len(self.queue)
            self.add(h, front=True)
            added += len(self.queue) > before
        log(f"LLM предложила {added} новых гипотез")
        if added == 0:
            self.empty_llm_rounds += 1
            if self.empty_llm_rounds >= 2:
                self.max_llm_rounds = self.llm_rounds  # новых идей нет — не тратим время на повторные запросы
        return added

    def normalize(self, item):
        if not isinstance(item, dict):
            return None
        act = item.get("action") if isinstance(item.get("action"), dict) else item
        t = str(act.get("type", "")).lower().strip()
        t = {"custom_feature": "custom_features", "custom_code": "custom_features", "feature_code": "custom_features",
             "code": "custom_features", "features": "custom_features", "lgb_params": "params",
             "hyperparameters": "params", "lightgbm_params": "params", "drop": "drop_features",
             "freq_encoding": "frequency_encoding", "count_encoding": "frequency_encoding"}.get(t, t)
        title = short(str(item.get("title") or t), 120)
        base = {"title": title, "source": "llm", "rationale": short(str(item.get("rationale", "")), 300)}
        if t == "custom_features" and isinstance(act.get("code"), str):
            code = act["code"].strip()
            m = re.fullmatch(r"BLOCK_?(\d+)", code, flags=re.I)
            if m:
                i = int(m.group(1)) - 1
                blocks = getattr(self, "code_blocks", [])
                code = blocks[i] if 0 <= i < len(blocks) else ""
            return {**base, "kind": "custom", "args": {"code": code}} if "make_features" in code else None
        if t == "params":
            raw = act.get("params") if isinstance(act.get("params"), dict) else {
                k: v for k, v in act.items() if k not in ("type",)}
            p = clean_params(raw)
            return {**base, "kind": "params", "args": {"params": p}} if p else None
        if t == "drop_features" and isinstance(act.get("columns"), list):
            return {**base, "kind": "drop_features", "args": {"columns": [str(c) for c in act["columns"]][:50]}}
        if t == "frequency_encoding" and isinstance(act.get("columns"), list):
            return {**base, "kind": "freq", "args": {"cols": [str(c) for c in act["columns"]][:30]}}
        if t == "group_deviation" and isinstance(act.get("pairs"), list):
            pairs = [tuple(map(str, p[:2])) for p in act["pairs"] if isinstance(p, (list, tuple)) and len(p) >= 2]
            return {**base, "kind": "group_dev", "args": {"pairs": pairs[:12]}} if pairs else None
        if t == "objective":
            v = str(act.get("objective", "l2")).lower()
            return {**base, "kind": "objective", "args": {"variant": v if v in ("huber", "fair") else None}}
        if t == "target_transform":
            return {**base, "kind": "target_transform", "args": {"how": str(act.get("transform", "none")).lower()}}
        if t == "class_weight":
            try:
                return {**base, "kind": "class_weight", "args": {"power": round(float(act.get("power", 0.5)), 2)}}
            except Exception:
                return None
        if t == "entity_history" and isinstance(act.get("key_columns"), list):
            return {**base, "kind": "entity_history", "args": {"key": [str(c) for c in act["key_columns"]]}}
        if t == "recency_weight":
            try:
                return {**base, "kind": "recency_weight", "args": {"strength": round(float(act.get("strength", 1.0)), 2)}}
            except Exception:
                return None
        return None

    # ------------------------------------------------------------ главный цикл

    def run(self, deadline: float, max_experiments: int = 60, on_baseline=None):
        """deadline — момент (в секундах от старта), к которому исследование должно закончиться."""
        self.baseline(on_baseline)
        if self.best_score is None:
            return
        self.builtin()
        llm_ok = self.llm is not None and self.llm.enabled
        t_start = elapsed()
        n_exp = 0
        while n_exp < max_experiments:
            if elapsed() + self.est_eval_time() > deadline:
                log("исследование: время на эксперименты исчерпано")
                break
            if self.llm_rounds == 0:
                # первый раунд — после встроенных гипотез (но не позже середины исследования): LLM достраивает
                # поверх пути, который агент проходит и без неё, а не закрывает его своими ранними находками
                due = (not any(q.get("source") == "builtin" for q in self.queue)
                       or elapsed() - t_start > 0.5 * (deadline - t_start))
            else:
                due = self.since_llm >= 6
            want_llm = llm_ok and self.llm.enabled and self.llm_rounds < self.max_llm_rounds and (
                not self.queue or due)
            if want_llm and elapsed() + self.est_eval_time() + 120 < deadline:
                try:
                    self.llm_round()
                except Exception:
                    log_exc("llm_round")
            if not self.queue and self.retry:
                self.queue, self.retry = self.retry, []
            if not self.queue:
                h = self.search_hypothesis()
                if h is None:
                    break
                self.queue.append(h)
            h = self.queue.pop(0)
            ok = self.try_one(h)
            if h.get("source") == "search":
                self.search_fails = 0 if ok else self.search_fails + 1
            n_exp += 1
            self.since_llm += 1
            self.save()
        self.save()

    def search_hypothesis(self):
        """Когда идеи кончились: случайный шаг по параметрам вокруг текущей лучшей конфигурации."""
        if self.search_tries >= 15 or self.search_fails >= 6:
            return None
        rng = np.random.default_rng(1000 + self.search_tries)
        self.search_tries += 1
        cur = {"num_leaves": 31, "min_child_samples": 40, "feature_fraction": 0.7, "bagging_fraction": 0.8,
               "lambda_l2": 1.0, **self.cfg.lgb_params}
        moves = {
            "num_leaves": lambda v: int(round(v * rng.choice([0.5, 0.7, 1.5, 2.0]))),
            "min_child_samples": lambda v: int(round(v * rng.choice([0.4, 0.7, 1.5, 2.5]))),
            "feature_fraction": lambda v: float(np.clip(v + rng.choice([-0.2, -0.1, 0.1, 0.2]), 0.15, 1.0)),
            "bagging_fraction": lambda v: float(np.clip(v + rng.choice([-0.2, -0.1, 0.1]), 0.4, 1.0)),
            "lambda_l2": lambda v: float(max(v, 0.1) * rng.choice([0.2, 0.5, 3.0, 10.0])),
            "min_split_gain": lambda v: float(rng.choice([0.0, 0.01, 0.1])),
            "path_smooth": lambda v: float(rng.choice([0.0, 1.0, 10.0])),
            "max_bin": lambda v: int(rng.choice([63, 127, 255, 511])),
        }
        for _ in range(20):
            keys = rng.choice(list(moves), size=int(rng.integers(1, 3)), replace=False)
            new = {k: moves[k](cur.get(k, 0.0)) for k in keys}
            new = clean_params(new)
            if any(cur.get(k) != v for k, v in new.items()):
                h = {"kind": "params", "args": {"params": new}, "source": "search",
                     "title": "поиск параметров: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                                               for k, v in new.items())}
                key = json.dumps({"kind": "params", "args": h["args"]}, sort_keys=True, default=str)
                if key not in self.tried:
                    self.tried.add(key)
                    return h
        return None

    def save(self):
        save_json(self.work / "journal.json", {"best_score": self.best_score, "metric": self.spec.metric,
                                               "config": self.cfg.to_dict(), "blocks": list(self.blocks),
                                               "journal": self.journal})
