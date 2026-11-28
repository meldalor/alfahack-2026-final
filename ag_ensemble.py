"""Смешивание моделей по OOF, решающее правило и поправка на сдвиг априорных вероятностей классов."""

import time

import numpy as np
import pandas as pd

from ag_metrics import (PROBA_METRICS, RANK_METRICS, TUNABLE, decide, direction, label_score, oof_score,
                        optimize_decision, proba_score, score)
from ag_utils import log


def _blend_metric(spec):
    """Метрика для подбора весов смеси: для метрик по меткам — logloss (гладкая и быстрая)."""
    if spec.task_type == "regression" or spec.metric in PROBA_METRICS:
        return spec.metric
    return "logloss"


def _eval(spec, metric, y, p, plan=None, mask=None):
    if spec.task_type == "regression":
        return score(metric, "regression", y[mask] if mask is not None else y, p[mask] if mask is not None else p)
    if plan is not None and metric in RANK_METRICS and mask is not None:
        return oof_score(metric, spec.task_type, y, p, plan=plan, mask=mask)
    yy = y[mask] if mask is not None else y
    pp = p[mask] if mask is not None else p
    return proba_score(metric, yy, pp) if metric in PROBA_METRICS else score(metric, spec.task_type, yy, pp)


def _simplex_reg_weights(M: np.ndarray, y: np.ndarray, metric: str, init_w=None,
                         time_budget: float = 4.0) -> np.ndarray:
    """Оптимизация весов на неотрицательном симплексе (w >= 0, sum(w) = 1) напрямую под целевую метрику регрессии."""
    from scipy.optimize import minimize

    k = M.shape[1]
    if k == 1:
        return np.ones(1, dtype=float)
    sign = 1.0 if direction(metric) == "max" else -1.0
    deadline = time.time() + float(time_budget)

    class _StopOpt(Exception):
        pass

    def _softmax(z):
        z = np.asarray(z, dtype=float)
        z = z - np.max(z)
        e = np.exp(np.clip(z, -40.0, 0.0))
        return e / np.sum(e)

    best_w = np.asarray(init_w, dtype=float) if init_w is not None else np.full(k, 1.0 / k)
    best_w = np.clip(best_w, 0.0, None)
    best_w = best_w / max(best_w.sum(), 1e-12)
    best_val = -sign * score(metric, "regression", y, M @ best_w)

    def _loss(z):
        nonlocal best_val, best_w
        if time.time() > deadline:
            raise _StopOpt()
        w = _softmax(z)
        v = -sign * score(metric, "regression", y, M @ w)
        if np.isfinite(v) and v < best_val - 1e-12:
            best_val = float(v)
            best_w = w.copy()
        return v

    starts = [np.zeros(k)]
    if init_w is not None and len(init_w) == k:
        starts.append(np.log(np.clip(np.asarray(init_w, dtype=float), 1e-3, 1.0)))
    indiv = [sign * score(metric, "regression", y, M[:, j]) for j in range(k)]
    best_j = int(np.argmax(indiv))
    z_best = np.full(k, -2.5)
    z_best[best_j] = 1.5
    starts.append(z_best)

    n = len(y)
    maxfev_p = 300 if n > 50_000 else 1500
    maxfev_n = 400 if n > 50_000 else 1500
    for z0 in starts:
        if time.time() > deadline:
            break
        for meth, maxit, maxfev in (("Powell", 400, maxfev_p), ("Nelder-Mead", 800, maxfev_n)):
            if time.time() > deadline:
                break
            try:
                res = minimize(_loss, z0, method=meth, options={"maxiter": maxit, "maxfev": maxfev})
                if np.isfinite(res.fun) and res.fun < best_val - 1e-12:
                    best_val = float(res.fun)
                    best_w = _softmax(res.x)
                    z0 = res.x
            except _StopOpt:
                break
            except Exception:
                continue
    best_w = np.where(best_w < 0.005, 0.0, best_w)
    s = best_w.sum()
    return (best_w / s) if s > 0 else np.full(k, 1.0 / k)


def _simplex_logloss_weights(P_list: list[np.ndarray], y: np.ndarray, init_w=None) -> np.ndarray:
    """Точная быстрая оптимизация весов на неотрицательном симплексе для logloss (аналитический градиент L-BFGS-B)."""
    from scipy.optimize import minimize

    k = len(P_list)
    if k == 1:
        return np.ones(1, dtype=float)
    y_arr = np.asarray(y, dtype=np.int64)
    rows = np.arange(len(y_arr))
    Py = np.column_stack([np.clip(np.asarray(p, dtype=np.float64)[rows, y_arr], 1e-12, 1.0) for p in P_list])

    def _softmax(z):
        z = np.asarray(z, dtype=float)
        z = z - np.max(z)
        e = np.exp(np.clip(z, -40.0, 0.0))
        return e / np.sum(e)

    def _nll_and_grad(z):
        w = _softmax(z)
        py = np.clip(Py @ w, 1e-12, 1.0)
        loss = -float(np.mean(np.log(py)))
        gw = -np.mean(Py / py[:, None], axis=0)
        gz = w * (gw - float(np.dot(w, gw)))
        return loss, gz

    starts = [np.zeros(k)]
    if init_w is not None and len(init_w) == k:
        starts.append(np.log(np.clip(np.asarray(init_w, dtype=float), 1e-3, 1.0)))
    best_w = np.asarray(init_w, dtype=float) if init_w is not None else np.full(k, 1.0 / k)
    best_w = np.clip(best_w, 0.0, None)
    best_w = best_w / max(best_w.sum(), 1e-12)
    best_val = -float(np.mean(np.log(np.clip(Py @ best_w, 1e-12, 1.0))))
    for z0 in starts:
        try:
            res = minimize(_nll_and_grad, z0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
            if np.isfinite(res.fun) and res.fun < best_val - 1e-12:
                best_val = float(res.fun)
                best_w = _softmax(res.x)
        except Exception:
            continue
    best_w = np.where(best_w < 0.005, 0.0, best_w)
    s = best_w.sum()
    return (best_w / s) if s > 0 else np.full(k, 1.0 / k)


def _mix_cls_proba(probs: list[np.ndarray], w: np.ndarray, log_odds: bool = False) -> np.ndarray:
    if not log_odds:
        out = sum(float(wi) * p for wi, p in zip(w, probs) if wi > 0)
        return out
    eps = 1e-6
    logits = sum(float(wi) * np.log(np.clip(p, eps, 1.0)) for wi, p in zip(w, probs) if wi > 0)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(np.clip(logits, -40.0, 0.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _fast_bin_rank_score(metric: str, y01: np.ndarray, s: np.ndarray) -> float:
    """Exact vectorized binary PR-AUC / ROC-AUC matching sklearn with tie grouping (~5x faster inside blend loops)."""
    y = np.asarray(y01, dtype=np.int8)
    n = len(y)
    if n < 2:
        return float("nan")
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = np.asarray(s, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y[order]
    distinct = np.flatnonzero(np.diff(s_sorted) != 0)
    idx = np.append(distinct, n - 1)
    tps = np.cumsum(y_sorted, dtype=np.float64)[idx]
    fps = (idx + 1).astype(np.float64) - tps
    if metric == "average_precision":
        prec = tps / (tps + fps)
        rec_diff = np.diff(np.r_[0.0, tps]) / float(n_pos)
        return float(np.sum(prec * rec_diff))
    tpr = np.r_[0.0, tps / float(n_pos)]
    fpr = np.r_[0.0, fps / float(n_neg)]
    return float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr)))


def _simplex_cls_weights(oofs: list[np.ndarray], y: np.ndarray, metric: str, spec, plan=None, mask=None,
                         init_w=None, log_odds: bool = False) -> tuple[np.ndarray, float]:
    """Оптимизация весов классификации на неотрицательном симплексе (линейные вероятности или log-odds).
    Использует быструю точную векторизацию PR-AUC/ROC-AUC, покоординатный спуск по активным моделям,
    стратифицированную подвыборку для крупных RANK_METRICS (>25k) и финальную проверку на полном OOF."""
    from scipy.optimize import minimize

    k = len(oofs)
    sign = 1.0 if direction(metric) == "max" else -1.0
    if k == 1:
        p = _mix_cls_proba(oofs, np.ones(1), log_odds=log_odds)
        return np.ones(1, dtype=float), sign * _eval(spec, metric, y, p, plan=plan, mask=mask)

    y_arr = np.asarray(y)
    idx = np.arange(len(y_arr)) if mask is None else np.where(mask)[0]
    if len(idx) == 0:
        return np.full(k, 1.0 / k), float("nan")

    pos = {int(g): i for i, g in enumerate(idx)}
    fold_groups = None
    if plan is not None and metric in RANK_METRICS and len(getattr(plan, "folds", [])) >= 2:
        gs = []
        for _, va in plan.folds:
            g = np.array([pos[int(v)] for v in va if int(v) in pos], dtype=int)
            if len(g):
                gs.append(g)
        if gs:
            fold_groups = gs

    y_m = y_arr[idx]
    eps = 1e-6
    n_classes = int(oofs[0].shape[1])
    fast_bin = bool(n_classes == 2 and metric in RANK_METRICS)
    mats_full = [np.log(np.clip(o[idx], eps, 1.0)) if log_odds else o[idx] for o in oofs]
    if fast_bin and not log_odds:
        mats_s_full = [m[:, 1].astype(np.float64, copy=False) for m in mats_full]
    elif fast_bin and log_odds:
        mats_s_full = [(m[:, 1] - m[:, 0]).astype(np.float64, copy=False) for m in mats_full]
    else:
        mats_s_full = None

    def _mix_pre(mats, w):
        if not log_odds:
            out = None
            for wi, m in zip(w, mats):
                if wi > 0:
                    out = (wi * m) if out is None else (out + wi * m)
            return out
        logits = None
        for wi, m in zip(w, mats):
            if wi > 0:
                logits = (wi * m) if logits is None else (logits + wi * m)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(np.clip(logits, -40.0, 0.0))
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def _score_vec(vec, yy, groups):
        if groups is not None:
            vals = [_fast_bin_rank_score(metric, yy[g], vec[g]) for g in groups]
            vals = [v for v in vals if np.isfinite(v)]
            return (sign * float(np.mean(vals))) if vals else float("nan")
        v = _fast_bin_rank_score(metric, yy, vec)
        return (sign * float(v)) if np.isfinite(v) else float("nan")

    def _score_pre(mats, mats_s, yy, groups, w):
        if mats_s is not None:
            vec = None
            for wi, ms in zip(w, mats_s):
                if wi > 0:
                    vec = (wi * ms) if vec is None else (vec + wi * ms)
            return _score_vec(vec, yy, groups)
        p = _mix_pre(mats, w)
        if groups is not None:
            vals = [proba_score(metric, yy[g], p[g]) for g in groups]
            vals = [v for v in vals if np.isfinite(v)]
            return (sign * float(np.mean(vals))) if vals else float("nan")
        v = proba_score(metric, yy, p) if metric in PROBA_METRICS else score(metric, spec.task_type, yy, p)
        return (sign * float(v)) if np.isfinite(v) else float("nan")

    # Evaluate individual models + init_w to guarantee strong non-degenerate starting point
    indiv_s = np.array([_score_pre(mats_full, mats_s_full, y_m, fold_groups, np.eye(k)[j]) for j in range(k)], dtype=float)
    best_j = int(np.nanargmax(indiv_s)) if np.isfinite(indiv_s).any() else 0
    best_w = np.eye(k)[best_j]
    best_s = float(indiv_s[best_j]) if np.isfinite(indiv_s[best_j]) else float("-inf")
    if init_w is not None and len(init_w) == k:
        w0 = np.clip(np.asarray(init_w, dtype=float), 0.0, None)
        if w0.sum() > 0:
            w0 = w0 / w0.sum()
            s0 = _score_pre(mats_full, mats_s_full, y_m, fold_groups, w0)
            if np.isfinite(s0) and s0 > best_s + 1e-12:
                best_s, best_w = s0, w0

    # Для 2 моделей 1D-сетка + локальное уточнение даёт глобальный оптимум за <= 30 оценок (<0.1s на 65k строк)
    if k == 2:
        best_a = float(best_w[0])
        for a in np.linspace(0.0, 1.0, 21):
            s = _score_pre(mats_full, mats_s_full, y_m, fold_groups, np.array([a, 1.0 - a]))
            if np.isfinite(s) and (not np.isfinite(best_s) or s > best_s + 1e-12):
                best_s, best_a = s, float(a)
        lo, hi = max(0.0, best_a - 0.05), min(1.0, best_a + 0.05)
        for a in np.linspace(lo, hi, 11):
            s = _score_pre(mats_full, mats_s_full, y_m, fold_groups, np.array([a, 1.0 - a]))
            if np.isfinite(s) and (not np.isfinite(best_s) or s > best_s + 1e-12):
                best_s, best_a = s, float(a)
        best_w = np.array([best_a, 1.0 - best_a], dtype=float)
        best_w = np.where(best_w < 1e-3, 0.0, best_w)
        best_w = best_w / max(best_w.sum(), 1e-12)
        return best_w, float(_score_pre(mats_full, mats_s_full, y_m, fold_groups, best_w))

    # Stratified subsample for huge ranking datasets (>25k) preserving exact class proportions per fold
    big_rank = metric in RANK_METRICS and len(idx) > 25_000
    opt_mats, opt_mats_s, opt_y, opt_groups = mats_full, mats_s_full, y_m, fold_groups
    if big_rank:
        rng = np.random.default_rng(42)

        def _strat_pick(g_arr, cap):
            if len(g_arr) <= cap:
                return g_arr
            yg = y_m[g_arr].astype(int)
            parts = []
            for cls in np.unique(yg):
                ci = g_arr[yg == cls]
                kt = max(1, int(round(cap * len(ci) / len(g_arr))))
                parts.append(rng.choice(ci, size=min(len(ci), kt), replace=False) if len(ci) > kt else ci)
            out_g = np.concatenate(parts)
            out_g.sort()
            return out_g

        if fold_groups is not None:
            per_g = max(1000, 25_000 // len(fold_groups))
            sub_parts, new_groups = [], []
            off = 0
            for g in fold_groups:
                sg = _strat_pick(g, per_g)
                sub_parts.append(sg)
                new_groups.append(np.arange(off, off + len(sg), dtype=int))
                off += len(sg)
            sub = np.concatenate(sub_parts)
            opt_groups = new_groups
        else:
            sub = _strat_pick(np.arange(len(idx), dtype=int), 25_000)
            opt_groups = None
        opt_mats = [m[sub] for m in mats_full]
        opt_mats_s = [m[sub] for m in mats_s_full] if mats_s_full is not None else None
        opt_y = y_m[sub]

    # Monotonic coordinate line search on opt_* (fast O(k) sweeps; never collapses to uniform 1/k!)
    w_coord = best_w.copy()
    cur_opt_s = _score_pre(opt_mats, opt_mats_s, opt_y, opt_groups, w_coord)
    order_k = np.argsort(np.nan_to_num(indiv_s, nan=-1e9))[::-1]
    alphas_ls = (0.04, 0.08, 0.15, 0.25, 0.40, 0.60, 0.80)
    for _ in range(2):
        improved = False
        cur_vec = None
        if opt_mats_s is not None:
            for wi, ms in zip(w_coord, opt_mats_s):
                if wi > 0:
                    cur_vec = (wi * ms) if cur_vec is None else (cur_vec + wi * ms)
        for j in order_k:
            best_a, best_as = 0.0, cur_opt_s
            for a in alphas_ls:
                if opt_mats_s is not None and cur_vec is not None:
                    sc = _score_vec((1.0 - a) * cur_vec + a * opt_mats_s[j], opt_y, opt_groups)
                else:
                    w_try = (1.0 - a) * w_coord
                    w_try[j] += a
                    sc = _score_pre(opt_mats, opt_mats_s, opt_y, opt_groups, w_try)
                if np.isfinite(sc) and sc > best_as + 1e-12:
                    best_as, best_a = sc, a
            if best_a > 0.0:
                w_coord = (1.0 - best_a) * w_coord
                w_coord[j] += best_a
                cur_opt_s = best_as
                if opt_mats_s is not None and cur_vec is not None:
                    cur_vec = (1.0 - best_a) * cur_vec + best_a * opt_mats_s[j]
                improved = True
        if not improved:
            break

    cand_ws = [w_coord]
    # Active-set Powell refinement: optimize only over active + top-4 models so maxfev isn't wasted on dead models
    active = sorted(set(np.flatnonzero(w_coord >= 0.005).tolist()) | set(order_k[:min(4, k)].tolist()))
    if len(active) >= 2:
        act_mats = [opt_mats[j] for j in active]
        act_mats_s = [opt_mats_s[j] for j in active] if opt_mats_s is not None else None

        def _softmax(z):
            z = np.asarray(z, dtype=float)
            z = z - np.max(z)
            e = np.exp(np.clip(z, -40.0, 0.0))
            return e / np.sum(e)

        def _loss_act(z):
            s = _score_pre(act_mats, act_mats_s, opt_y, opt_groups, _softmax(z))
            return (-s) if np.isfinite(s) else 1e9

        w_act0 = w_coord[active]
        w_act0 = w_act0 / max(w_act0.sum(), 1e-12)
        starts = [np.log(np.clip(w_act0, 1e-3, 1.0))]
        if not big_rank:
            starts.append(np.zeros(len(active)))
        maxfev = 160 if big_rank else (220 if len(idx) > 20_000 else 350)
        for z0 in starts:
            try:
                res = minimize(_loss_act, z0, method="Powell", options={"maxfev": maxfev})
                if np.isfinite(res.fun):
                    w_full = np.zeros(k, dtype=float)
                    w_full[active] = _softmax(res.x)
                    cand_ws.append(w_full)
            except Exception:
                continue

    for cw in cand_ws:
        cw = np.where(cw < 1e-3, 0.0, cw)
        sw = cw.sum()
        if sw <= 0:
            continue
        cw = cw / sw
        s_full = _score_pre(mats_full, mats_s_full, y_m, fold_groups, cw)
        if np.isfinite(s_full) and (not np.isfinite(best_s) or s_full > best_s + 1e-12):
            best_s, best_w = s_full, cw
    best_w = np.where(best_w < 1e-3, 0.0, best_w)
    s = best_w.sum()
    best_w = (best_w / s) if s > 0 else np.full(k, 1.0 / k)
    return best_w, float(_score_pre(mats_full, mats_s_full, y_m, fold_groups, best_w))


def blend(results, y, mask, spec, max_steps: int = 30, plan=None, allow_log_odds: bool = True):
    """Жадный отбор с возвращением (Caruana) + доуточнение на неотрицательном симплексе (и log-odds для классификации)."""
    usable = [r for r in results if r is not None and r.test is not None
              and not np.isnan(r.oof[mask]).any() and np.isfinite(r.score)]
    if not usable:
        return None
    metric = _blend_metric(spec)
    sign = 1 if direction(metric) == "max" else -1
    use_plan = plan if (plan is not None and metric in RANK_METRICS and len(getattr(plan, "folds", [])) >= 2) else None
    scores = [sign * _eval(spec, metric, y, r.oof, plan=use_plan, mask=mask) for r in usable]
    order = np.argsort(scores)[::-1]
    counts = np.zeros(len(usable))
    counts[order[0]] = 1
    cur_full = usable[order[0]].oof.copy()
    cur_s = scores[order[0]]
    for _ in range(1, max_steps):
        best_j, best_s, best_p = None, cur_s, None
        n = counts.sum()
        for j, r in enumerate(usable):
            p = (cur_full * n + r.oof) / (n + 1)
            s = sign * _eval(spec, metric, y, p, plan=use_plan, mask=mask)
            if s > best_s + 1e-12:
                best_j, best_s, best_p = j, s, p
        if best_j is None:
            break
        counts[best_j] += 1
        cur_full, cur_s = best_p, best_s
    w = counts / counts.sum()
    use_log_odds = False
    if spec.task_type == "regression" and len(usable) >= 2:
        M = np.column_stack([np.asarray(r.oof[mask], dtype=float).reshape(-1) for r in usable])
        w_sx = _simplex_reg_weights(M, np.asarray(y)[mask], metric, init_w=w)
        s_sx = sign * score(metric, "regression", np.asarray(y)[mask], M @ w_sx)
        if s_sx > cur_s + 1e-12:
            w = w_sx
    elif spec.task_type != "regression" and len(usable) >= 2 and len(usable) <= 24:
        oofs = [np.asarray(r.oof, dtype=float) for r in usable]
        w_lin, s_lin = _simplex_cls_weights(oofs, y, metric, spec, plan=use_plan, mask=mask, init_w=w, log_odds=False)
        if np.isfinite(s_lin) and s_lin > cur_s + 1e-12:
            w, cur_s, use_log_odds = w_lin, s_lin, False
        if allow_log_odds:
            w_lo, s_lo = _simplex_cls_weights(oofs, y, metric, spec, plan=use_plan, mask=mask, init_w=w, log_odds=True)
            if np.isfinite(s_lo) and s_lo > cur_s + 1e-12:
                w, cur_s, use_log_odds = w_lo, s_lo, True
    if spec.task_type == "regression":
        oof = sum(wi * r.oof for wi, r in zip(w, usable) if wi > 0)
        test = sum(wi * r.test for wi, r in zip(w, usable) if wi > 0)
    else:
        oof = _mix_cls_proba([np.asarray(r.oof, dtype=float) for r in usable], w, log_odds=use_log_odds)
        test = _mix_cls_proba([np.asarray(r.test, dtype=float) for r in usable], w, log_odds=use_log_odds)
    weights = {r.name: round(float(wi), 3) for wi, r in zip(w, usable) if wi >= 0.005}
    return {"weights": weights, "oof": oof, "test": test,
            "score": score(spec.metric, spec.task_type, y[mask], oof[mask]) if spec.task_type == "regression"
            else None}


def rank_blend(results, y, mask, spec, max_steps: int = 30, plan=None):
    """Ранговое смешивание для ранжирующих метрик (ROC-AUC, PR-AUC): выравнивает масштаб вероятностей разных семейств."""
    from scipy.stats import rankdata

    if spec.task_type != "binary" or spec.metric not in ("roc_auc", "average_precision"):
        return None
    usable = [r for r in results if r is not None and r.test is not None
              and not np.isnan(r.oof[mask]).any() and np.isfinite(r.score)]
    if len(usable) < 2:
        return None

    def to_rank_proba(oof, test):
        ro = np.full_like(oof, np.nan, dtype=float)
        if plan is not None and len(getattr(plan, "folds", [])) >= 2:
            for _, va in plan.folds:
                va_m = va[mask[va]]
                if len(va_m):
                    p1 = rankdata(oof[va_m, 1]) / (len(va_m) + 1.0)
                    ro[va_m, 1] = p1
                    ro[va_m, 0] = 1.0 - p1
        else:
            p1 = rankdata(oof[mask, 1]) / (int(mask.sum()) + 1.0)
            ro[mask, 1] = p1
            ro[mask, 0] = 1.0 - p1
        t1 = rankdata(test[:, 1]) / (len(test) + 1.0)
        rt = np.column_stack([1.0 - t1, t1])
        return ro, rt

    ranked = []
    for r in usable:
        ro, rt = to_rank_proba(r.oof, r.test)
        ranked.append(type(r)(name=r.name, oof=ro, test=rt, score=r.score, fold_scores=r.fold_scores,
                              iters=r.iters, importance=r.importance, seconds=r.seconds, config=r.config))
    return blend(ranked, y, mask, spec, max_steps=max_steps, plan=plan, allow_log_odds=False)


def stack_blend(results, y, mask, spec, plan=None):
    """Честный вложенный OOF-стекинг на неотрицательном симплексе (w >= 0, sum(w) = 1) напрямую под целевую
    метрику для регрессии (MAE, RMSLE, MAPE, RMSE) и классификации (PR-AUC, ROC-AUC, LogLoss).
    Оценка meta_oof строго out-of-fold по границам plan.folds (>= 2 фолдов) — без отрицательных весов."""
    from sklearn.model_selection import KFold, StratifiedKFold

    usable = [r for r in results if r is not None and r.test is not None
              and not np.isnan(r.oof[mask]).any() and np.isfinite(r.score)]
    idx = np.where(mask)[0]
    if len(usable) < 3 or len(idx) < 300:
        return None
    ym = np.asarray(y)[idx]

    # Строим честные вложенные фолды поверх idx строго по реальным границам plan.folds (поддержаны >= 2 фолдов)
    meta_folds = []
    if plan is not None and len(getattr(plan, "folds", [])) >= 2:
        pos = {int(g): i for i, g in enumerate(idx)}
        groups = []
        for _, va in plan.folds:
            g = np.array([pos[int(v)] for v in va if int(v) in pos], dtype=int)
            if len(g):
                groups.append(g)
        if len(groups) >= 2:
            all_i = np.arange(len(idx))
            for g in groups:
                va_m = np.zeros(len(idx), bool)
                va_m[g] = True
                tr_i = all_i[~va_m]
                if len(tr_i):
                    meta_folds.append((tr_i, g))
    if not meta_folds:
        if spec.task_type != "regression":
            cnt = np.bincount(ym.astype(int))
            min_c = int(cnt[cnt > 0].min()) if (cnt > 0).any() else 0
            if min_c >= 3:
                meta_folds = list(StratifiedKFold(min(5, min_c), shuffle=True, random_state=42).split(idx, ym))
        if not meta_folds:
            meta_folds = list(KFold(5, shuffle=True, random_state=42).split(idx))

    if spec.task_type == "regression":
        M_oof = np.column_stack([np.asarray(r.oof[idx], dtype=float).reshape(-1) for r in usable])
        M_test = np.column_stack([np.asarray(r.test, dtype=float).reshape(-1) for r in usable])
        if not (np.isfinite(M_oof).all() and np.isfinite(M_test).all()):
            return None
        y_lo, y_hi = float(np.nanmin(y)), float(np.nanmax(y))
        mo = np.full(len(y), np.nan, dtype=float)
        for tr, va in meta_folds:
            w_f = _simplex_reg_weights(M_oof[tr], ym[tr], spec.metric)
            mo[idx[va]] = np.clip(M_oof[va] @ w_f, y_lo, y_hi)
        w_full = _simplex_reg_weights(M_oof, ym, spec.metric)
        mt = np.clip(M_test @ w_full, y_lo, y_hi)
        weights = {r.name: round(float(c), 3) for r, c in zip(usable, w_full) if c >= 0.005}
        return {"weights": weights, "oof": mo, "test": mt,
                "score": score(spec.metric, spec.task_type, y[mask], mo[mask])}

    P_oof = [np.clip(np.asarray(r.oof[idx], dtype=float), 1e-7, 1.0) for r in usable]
    P_test = [np.clip(np.asarray(r.test, dtype=float), 1e-7, 1.0) for r in usable]
    if not all(np.isfinite(p).all() for p in P_oof + P_test):
        return None
    P_oof = [p / np.maximum(p.sum(1, keepdims=True), 1e-12) for p in P_oof]
    P_test = [p / np.maximum(p.sum(1, keepdims=True), 1e-12) for p in P_test]
    metric = _blend_metric(spec)
    n_classes = P_oof[0].shape[1]
    mo = np.full((len(y), n_classes), np.nan, dtype=float)

    def _fit_cls_w(p_list, yy, use_plan=None, use_mask=None):
        if metric == "logloss":
            return _simplex_logloss_weights(p_list, yy)
        w_s, _ = _simplex_cls_weights(p_list, yy, metric, spec, plan=use_plan, mask=use_mask, log_odds=False)
        w_s = np.where(w_s < 0.005, 0.0, w_s)
        sw = w_s.sum()
        return (w_s / sw) if sw > 0 else np.full(len(p_list), 1.0 / len(p_list))

    for tr, va in meta_folds:
        if len(np.unique(ym[tr])) < 2:
            return None
        w_f = _fit_cls_w([p[tr] for p in P_oof], ym[tr])
        p_va = sum(wi * p[va] for wi, p in zip(w_f, P_oof) if wi > 0)
        mo[idx[va]] = p_va / np.maximum(p_va.sum(1, keepdims=True), 1e-12)
    use_p = plan if (plan is not None and metric in RANK_METRICS and len(getattr(plan, "folds", [])) >= 2) else None
    if metric == "logloss":
        w_full = _simplex_logloss_weights(P_oof, ym)
    else:
        full_oofs = [np.asarray(r.oof, dtype=float) for r in usable]
        w_full, _ = _simplex_cls_weights(full_oofs, y, metric, spec, plan=use_p, mask=mask, log_odds=False)
        w_full = np.where(w_full < 0.005, 0.0, w_full)
        sw = w_full.sum()
        w_full = (w_full / sw) if sw > 0 else np.full(len(usable), 1.0 / len(usable))
    mt = sum(wi * p for wi, p in zip(w_full, P_test) if wi > 0)
    mt = mt / np.maximum(mt.sum(1, keepdims=True), 1e-12)
    weights = {r.name: round(float(c), 3) for r, c in zip(usable, w_full) if c >= 0.005}
    return {"weights": weights, "oof": mo, "test": mt, "score": None}


def em_test_priors(proba_test, train_priors, iters: int = 200):
    """Оценка априорных вероятностей классов на тесте по EM (Saerens et al., 2002)."""
    p = np.asarray(train_priors, float).copy()
    base = np.asarray(train_priors, float)
    for _ in range(iters):
        adj = proba_test * (p / base)[None, :]
        adj /= adj.sum(1, keepdims=True) + 1e-12
        new = adj.mean(0)
        if np.abs(new - p).max() < 1e-6:
            p = new
            break
        p = new
    return p


def _em_priors(p, prior, iters: int = 500, tol: float = 1e-7, alpha: float = 30.0):
    """EM для долей классов на новой выборке (Saerens et al., 2002); классы с нулевой долей в train не трогаем.

    alpha — псевдострок с долями train (MAP-оценка): в маленькой группе EM иначе подгоняется под шум."""
    act = prior > 0
    q = prior.copy()
    inv = np.where(act, 1.0 / np.where(act, prior, 1.0), 0.0)
    n = len(p)
    for _ in range(iters):
        r = np.where(act, q * inv, 1.0)
        w = p * r[None, :]
        s = w.sum(1, keepdims=True)
        w /= np.where(s > 0, s, 1.0)
        qn = np.where(act, (w.sum(0) + alpha * prior) / (n + alpha), 0.0)
        qn = qn / max(qn.sum(), 1e-12)
        if np.abs(qn - q).max() < tol:
            return qn
        q = qn
    return q


class LabelShift:
    """Сдвиг долей классов между train и test, оценённый отдельно в группах похожих строк.

    Группы — листья неглубокого дерева решений по исходным признакам (train -> класс). Внутри группы считается,
    что p(x | класс) в train и test один и тот же, а доли классов могут отличаться; их оценивает EM по
    вероятностям модели на тесте. Глобальная EM ошибается, когда сдвиг в разных группах разный: в UNSW-NB15 в
    тесте больше «нормального» трафика с TTL атакующих машин, и глобальная EM вместо Normal увеличивала Fuzzers.
    Классы, которые модель сильно путает между собой, оцениваются вместе: по вероятностям их доли неразличимы,
    и EM разносит их в крайности (обнуляла DoS); внутри такой группы классов сохраняется пропорция train.
    Оценка по одному дереву зависит от его случайной формы, поэтому разбиений несколько (деревья разной
    величины) и поправки усредняются.

    apply() пересчитывает вероятности OOF и теста под оценённые доли и даёт веса строк OOF: метрика по OOF с
    этими весами — оценка метрики на тесте, по ней и подбирается решающее правило.
    """

    LEAVES = (3, 4, 6, 8, 12, 16)
    LEAF_FRACS = (0.02, 0.05)

    def __init__(self, X, Xt, y, mask, n_classes: int, merge_confusion: float = 0.3, seed: int = 0):
        from sklearn.tree import DecisionTreeClassifier

        self.k = n_classes
        self.y = np.asarray(y)
        self.mask = np.asarray(mask, bool)
        self.merge_confusion = merge_confusion
        self.active = False
        self.info = {}
        self._col_maps = {}
        idx = np.where(self.mask)[0]
        if len(idx) > 200_000:
            idx = np.sort(np.random.default_rng(seed).choice(idx, 200_000, replace=False))
        A = self._numeric(X.iloc[idx], fit=True)
        trees, seen = [], set()
        for leaves in self.LEAVES:
            for frac in self.LEAF_FRACS:
                tree = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=max(50, int(frac * len(idx))),
                                              random_state=seed)
                tree.fit(A, self.y[idx])
                sig = (tree.get_n_leaves(), hash(tree.apply(A[:5000]).tobytes()))
                if sig in seen:  # то же разбиение (ограничение размера листа не сработало)
                    continue
                seen.add(sig)
                trees.append(tree)
        del A
        lt, le = self._leaves(trees, X), self._leaves(trees, Xt)
        self.splits = [(lt[:, i], le[:, i]) for i in range(len(trees))]

    def _leaves(self, trees, df, chunk: int = 250_000):
        """Номера листьев всех деревьев для строк df; порциями, чтобы не держать всю таблицу числами."""
        out = np.zeros((len(df), len(trees)), np.int32)
        for i in range(0, len(df), chunk):
            a = self._numeric(df.iloc[i:i + chunk], fit=False)
            for j, t in enumerate(trees):
                out[i:i + chunk, j] = t.apply(a)
        return out

    def _numeric(self, df, fit: bool = False):
        cols = []
        for c in df.columns:
            s = df[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                cols.append(s.cat.codes.to_numpy(np.float32))
            elif pd.api.types.is_numeric_dtype(s):
                cols.append(s.to_numpy(np.float32, na_value=np.nan))
            else:
                ss = s.astype("string")
                if fit or c not in self._col_maps:
                    codes, uniques = pd.factorize(ss, sort=False)
                    self._col_maps[c] = {v: i for i, v in enumerate(uniques)}
                    cols.append(codes.astype(np.float32))
                else:
                    cols.append(ss.map(self._col_maps[c]).fillna(-1.0).to_numpy(np.float32))
        a = np.column_stack(cols) if cols else np.zeros((len(df), 1), np.float32)
        a = np.where(np.isfinite(a), a, -1e9)
        return a.astype(np.float32)

    def _superclasses(self, oof):
        k = self.k
        m = self.mask & ~np.isnan(oof).reshape(len(oof), -1).any(1)
        pred = oof[m].argmax(1)
        yy = self.y[m]
        conf = np.zeros((k, k))
        for i in range(k):
            r = yy == i
            if r.any():
                conf[i] = np.bincount(pred[r], minlength=k) / r.sum()
        sym = conf + conf.T
        parent = list(range(k))

        def root(a):
            while parent[a] != a:
                a = parent[a]
            return a

        if k >= 3:  # в бинарной задаче объединять нечего: EM с одной долей на группу корректна
            for i in range(k):
                for j in range(i + 1, k):
                    if sym[i, j] > self.merge_confusion:
                        parent[root(i)] = root(j)
        groups = np.array([root(i) for i in range(k)])
        if len(np.unique(groups)) < 2:
            groups = np.arange(k)  # всё слилось в одну группу — оценивать нечего, лучше без объединения
        ids = np.unique(groups)
        M = (groups[:, None] == ids[None, :]).astype(float)
        return M, [[int(i) for i in np.where(groups == g)[0]] for g in ids if (groups == g).sum() > 1]

    def _estimate(self, M, valid, probs, leaves_of, sub=None):
        """Поправки долей по группам для строк с вероятностями probs и разбиениями leaves_of[i].

        Возвращает (r_tr, r_target, оценка числа строк по классам, средний по строкам сдвиг долей в группах)."""
        k, n_t = self.k, len(probs)
        r_tr = np.zeros((len(self.y), k))
        r_t = np.zeros((n_t, k))
        est = np.zeros(k)
        shift_mass = 0.0
        for (leaf_tr, _), leaf_t in zip(self.splits, leaves_of):
            rs_tr = np.ones((len(self.y), k))
            rs_t = np.ones((n_t, k))
            for g in np.unique(leaf_t):
                t_rows = leaf_t == g
                tr_rows = (leaf_tr == g) & valid
                n_g = int(t_rows.sum())
                pri = np.bincount(self.y[tr_rows].astype(int), minlength=k) / max(1, tr_rows.sum())
                if n_g < 30 or tr_rows.sum() < 30:
                    est += pri * n_g
                    continue
                pri_s = pri @ M
                em_rows = t_rows & sub if (sub is not None and (t_rows & sub).sum() >= 30) else t_rows
                q_s = _em_priors(probs[em_rows] @ M, pri_s)
                ratio = M @ np.where(pri_s > 0, q_s / np.where(pri_s > 0, pri_s, 1.0), 1.0)
                ratio = np.clip(np.where(pri > 0, ratio, 1.0), 0.05, 20.0)
                rs_t[t_rows] = ratio
                rs_tr[leaf_tr == g] = ratio
                q = pri * ratio
                q = q / max(q.sum(), 1e-12)
                est += q * n_g
                shift_mass += 0.5 * np.abs(q - pri).sum() * n_g
            r_tr += rs_tr / len(self.splits)
            r_t += rs_t / len(self.splits)
        return r_tr, r_t, est / len(self.splits), shift_mass / max(n_t, 1) / len(self.splits)

    def fit(self, oof, test):
        """Оценка долей по вероятностям одной модели; дальше те же доли для всех кандидатов.

        Сдвиг учитывается, только если он заметно больше «нулевого» уровня — той же оценки по самим OOF
        (там сдвига нет, и всё найденное — шум групп и калибровки)."""
        valid = self.mask & ~np.isnan(oof).reshape(len(oof), -1).any(1)
        M, merged = self._superclasses(oof)
        # доли — средние по строкам: на большом тесте для EM хватает подвыборки
        sub = np.zeros(len(test), bool)
        sub[np.random.default_rng(0).choice(len(test), min(len(test), 100_000), replace=False)] = True
        self.r_tr, self.r_te, est, shift = self._estimate(M, valid, test, [le for _, le in self.splits], sub)
        idx = np.where(valid)[0]
        if len(idx) > 100_000:
            idx = np.sort(np.random.default_rng(1).choice(idx, 100_000, replace=False))
        _, _, _, null = self._estimate(M, valid, oof[idx], [lt[idx] for lt, _ in self.splits])
        self.active = bool(shift >= max(0.05, 2.0 * null + 0.02))
        self.info = {"within_group_shift": round(float(shift), 4), "null_shift": round(float(null), 4),
                     "splits": len(self.splits), "merged_classes": merged,
                     "test_counts_est": [int(round(v)) for v in est], "applied": self.active}
        log(f"сдвиг долей классов внутри групп: {shift:.3f} при нулевом уровне {null:.3f} "
            f"({'учитывается' if self.active else 'не учитывается'}); оценка числа строк теста по классам "
            f"{self.info['test_counts_est']}; совместно оценены {merged}")
        return self

    def apply(self, oof, test=None):
        """(вероятности OOF, вероятности теста, веса строк OOF) под оценённые доли классов в тесте."""
        a = oof * self.r_tr
        s = a.sum(1, keepdims=True)
        a = a / np.where(np.isfinite(s) & (s > 0), s, 1.0)
        b = None
        if test is not None:
            b = test * self.r_te
            b = b / np.maximum(b.sum(1, keepdims=True), 1e-12)
        yi = np.clip(self.y.astype(int), 0, self.k - 1)
        sw = self.r_tr[np.arange(len(yi)), yi]
        return a, b, sw


SHRINK_POWERS = (0.0, 0.25, 0.5, 0.75, 1.0)


def choose_shrink_power(spec, y, mask, oof, adv=None, adv_auc=None, shift: LabelShift | None = None):
    """Насколько доверять весам классов, подобранным по OOF: веса в степени p (0 — argmax, 1 — полные).

    Веса подбираются на 80% строк и проверяются на отложенных 20%: случайных и — если тест заметно
    отличается от train — самых похожих на тест (по adversarial validation). Полные веса, подобранные
    на тех же OOF, на которых оцениваются, переобучаются; на сдвинутом тесте — тем более.
    При учтённом сдвиге долей классов (shift) степень выбирается и по пересчитанным вероятностям со взвешенными
    строками, и по исходным, и берётся меньшая: оценка «под тест» зависит от точности оценки сдвига
    (на синтетике с 25 классами она выбирала полные веса, а на тесте лучше были сжатые).
    """
    n_classes = oof.shape[1]
    if spec.metric in PROBA_METRICS or spec.metric not in TUNABLE or n_classes <= 2:
        return 1.0, {}
    variants = [(oof, None)]
    if shift is not None and shift.active:
        a, _, sw_all = shift.apply(oof)
        variants.append((a, sw_all))
    chosen, details = [], {}
    for k, (pp_all, sw_all) in enumerate(variants):
        idx = np.where(mask & ~np.isnan(pp_all).reshape(len(pp_all), -1).any(1))[0]
        if len(idx) < 200:
            return 1.0, {}
        yy, pp = y[idx], pp_all[idx]
        ww = sw_all[idx] if sw_all is not None else None
        holds = [np.random.default_rng(0).random(len(idx)) < 0.2]
        if adv is not None and adv_auc is not None and adv_auc >= 0.65:
            a_ = adv[idx]
            holds.append(a_ >= np.quantile(a_, 0.8))
        totals = np.zeros(len(SHRINK_POWERS))
        for hold in holds:
            d = optimize_decision(spec.metric, yy[~hold], pp[~hold], sample_weight=None if ww is None else ww[~hold])
            w = np.asarray(d.get("weights", np.ones(n_classes)), float)
            for i, p in enumerate(SHRINK_POWERS):
                totals[i] += label_score(spec.metric, yy[hold], decide(pp[hold], spec.metric, {"weights": w ** p}),
                                         n_classes, None if ww is None else ww[hold])
        totals /= len(holds)
        chosen.append(SHRINK_POWERS[int(np.argmax(totals))])
        details["под доли теста" if k else "по OOF"] = {str(p): round(float(t), 5) for p, t in zip(SHRINK_POWERS, totals)}
    best = min(chosen)
    log(f"сжатие весов классов: выбрана степень {best} по отложенным частям {details}")
    return best, details


# метрики по матрице ошибок, для которых правило подбирается и по ожидаемой матрице на самом тесте
TEST_RULE_METRICS = ("f1_macro", "f1_weighted", "balanced_accuracy")


def _cm_score(metric: str, cm: np.ndarray) -> float:
    tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    if metric == "balanced_accuracy":
        ok = sup > 1e-9
        return float(np.mean(tp[ok] / sup[ok])) if ok.any() else 0.0
    f1 = 2 * tp / np.maximum(sup + pred, 1e-12)
    if metric == "f1_weighted":
        return float((f1 * sup).sum() / max(float(sup.sum()), 1e-12))
    return float(f1.mean())


def _expected_cm(p: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Ожидаемая матрица ошибок: строки — истинный класс с вероятностями p, столбцы — предсказанный lab."""
    k = p.shape[1]
    cm = np.zeros((k, k))
    for j in range(k):
        s = lab == j
        if s.any():
            cm[:, j] = p[s].sum(0)
    return cm


def calibrate(p, y, w=None, max_rows: int = 100_000):
    """Температура и сдвиги логитов классов по взвешенному NLL на OOF -> (функция пересчёта вероятностей, T)."""
    from scipy.optimize import minimize

    k = p.shape[1]
    idx = np.arange(len(y))
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_rows, replace=False))
    L = np.log(np.clip(p[idx], 1e-9, 1.0))
    yy = np.asarray(y)[idx].astype(int)
    ww = np.ones(len(idx)) if w is None else np.asarray(w, float)[idx]
    ww = ww / ww.sum()
    rows = np.arange(len(idx))

    def nll(th):
        t = np.exp(th[0])
        z = L / t + np.r_[0.0, th[1:]][None, :]
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        s = e.sum(1)
        g = e / s[:, None]
        lp = z[rows, yy] - np.log(s)
        g[rows, yy] -= 1.0
        g *= ww[:, None]
        return -float((ww * lp).sum()), np.r_[float((g * (-L / t)).sum()), g.sum(0)[1:]]

    th = minimize(nll, np.zeros(k), jac=True, method="L-BFGS-B",
                  bounds=[(-1.5, 1.5)] + [(-3.0, 3.0)] * (k - 1)).x

    def apply(q):
        z = np.log(np.clip(q, 1e-9, 1.0)) / np.exp(th[0]) + np.r_[0.0, th[1:]][None, :]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    return apply, float(np.exp(th[0]))


def expected_rule(metric: str, p: np.ndarray, max_rows: int = 100_000):
    """Веса классов, при которых ожидаемая метрика на строках с вероятностями p максимальна (без меток)."""
    k = p.shape[1]
    max_rows = min(max_rows, max(20_000, 2_000_000 // k))  # при многих классах — меньше строк: перебор ~ строки × классы²
    if len(p) > max_rows:
        p = p[np.sort(np.random.default_rng(0).choice(len(p), max_rows, replace=False))]
    grid = np.exp(np.linspace(-2.5, 2.5, 21 if k <= 30 else 9))
    w = np.ones(k)
    best = _cm_score(metric, _expected_cm(p, p.argmax(1)))
    for _ in range(3 if k <= 30 else 1):
        improved = False
        for j in range(k):
            keep = w[j]
            for g in grid:
                w[j] = g
                s = _cm_score(metric, _expected_cm(p, np.argmax(p * w[None, :], 1)))
                if s > best + 1e-9:
                    best, keep, improved = s, g, True
            w[j] = keep
        if not improved:
            break
    return w, best


def decide_on_test(spec, y, mask, oof, test_adj, decision, shift: LabelShift | None = None, adv_auc=None,
                   adv=None):
    """Правило под распределение самого теста, если тест заметно отличается от train.

    Веса классов, подобранные по OOF, оптимальны для распределения train (с поправкой на доли классов). Когда тест
    отличается и строками (adversarial AUC) — другие пропорции «трудных» строк, повторы одних и тех же векторов, —
    лучше правило, максимизирующее ожидаемую метрику на самих строках теста: вероятности модели как доли классов
    в ожидаемой матрице ошибок. Вероятности сначала калибруются по OOF (температура и сдвиги классов), строки
    train — с весами «похожести на тест» adv/(1 − adv) из adversarial validation: калибровка нужна там, где лежит
    тест. На UNSW-NB15: обучение на 80% train, меньше всего похожих на тест, оценка на 20% самых похожих — macro F1
    0.513 -> 0.546 (калибровка без весов похожести 0.537); то же 50/50 — 0.390 -> 0.480 (0.461); на случайных
    половинах OOF без сдвига 0.6211 -> 0.6209. Если по отложенным частям OOF веса классов не помогают (степень 0,
    `choose_shrink_power`), правило остаётся argmax: на синтетике, где в тесте меняется зависимость признака от
    класса, правило под тест там проигрывало (−0.008), а при степени больше нуля выигрывало (+0.003…+0.007).
    Возвращает (правило, вероятности теста) или None."""
    k = oof.shape[1]
    if (spec.metric not in TEST_RULE_METRICS or k < 3 or k > 100 or not decision or "weights" not in decision
            or len(test_adj) < 1000 or float(decision.get("weight_power", 1.0)) <= 0):
        return None
    active = shift is not None and shift.active
    if not (active or (adv_auc is not None and adv_auc >= 0.65)):
        return None
    a, sw = oof, None
    if active:
        a, _, sw = shift.apply(oof)
    m = np.asarray(mask, bool) & ~np.isnan(a).reshape(len(a), -1).any(1)
    w_cal = np.ones(len(a)) if sw is None else np.asarray(sw, float)
    if adv is not None and len(adv) == len(a):
        p_adv = np.clip(np.asarray(adv, float), 1e-6, 1 - 1e-6)
        w_cal = w_cal * np.clip(p_adv / (1 - p_adv) * len(a) / len(test_adj), 0.05, 20.0)
    cal, temp = calibrate(a[m], np.asarray(y)[m], w_cal[m])
    tc = cal(np.asarray(test_adj, float))
    w, exp_new = expected_rule(spec.metric, tc)
    old = decide(test_adj, spec.metric, decision)
    exp_old = _cm_score(spec.metric, _expected_cm(tc, old))
    if exp_new <= exp_old + 1e-6:
        return None
    new = np.argmax(tc * w[None, :], 1)
    log(f"правило под тест: калибровка T={temp:.3f}, ожидаемая {spec.metric} на тесте {exp_old:.4f} -> {exp_new:.4f}; "
        f"изменено меток {int((new != old).sum())}; классов было {np.bincount(old, minlength=k).tolist()}, "
        f"стало {np.bincount(new, minlength=k).tolist()}")
    return {**decision, "weights": w.tolist(), "rule": "test", "expected_test": round(exp_new, 5),
            "expected_test_oof_rule": round(exp_old, 5)}, tc


def apply_zero_mape_hurdle(X: pd.DataFrame, Xt: pd.DataFrame, y: np.ndarray, oof: np.ndarray, test: np.ndarray,
                           plan, mask: np.ndarray, n_jobs: int = 4):
    """Zero-aware MAPE hurdle: predicts P(y == 0 | x) strictly OOF and zeroes predictions above the optimal OOF threshold.
    Prevents 1e8-1e9 MAPE blowups on targets containing exact zeros (e.g. sales/claims)."""
    import lightgbm as lgb
    from ag_models import _sanitize_tree_frames, refit_rounds

    y = np.asarray(y, dtype=float)
    z = (np.abs(y) < 1e-12).astype(int)
    if z.mean() < 0.01 or z.mean() > 0.99:
        return None
    base_s = score("mape", "regression", y[mask], oof[mask])
    if not np.isfinite(base_s):
        return None
    X_s, Xt_s = _sanitize_tree_frames(X, Xt)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=30,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                  num_threads=max(1, min(8, n_jobs)), seed=42)
    pz_oof = np.full(len(y), np.nan, dtype=float)
    pz_test_parts = []
    iters = []
    for tr, va in plan.folds:
        if len(np.unique(z[tr])) < 2:
            return None
        dtr = lgb.Dataset(X_s.iloc[tr], z[tr], free_raw_data=True)
        dva = lgb.Dataset(X_s.iloc[va], z[va], reference=dtr)
        m = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
        bi = m.best_iteration or m.current_iteration()
        iters.append(bi)
        pz_oof[va] = m.predict(X_s.iloc[va], num_iteration=bi)
        if plan.mode == "cv":
            pz_test_parts.append(m.predict(Xt_s, num_iteration=bi))
    if plan.mode == "refit":
        n_round = refit_rounds(iters, plan, len(X_s))
        m = lgb.train(params, lgb.Dataset(X_s, z, free_raw_data=True), n_round)
        pz_test = m.predict(Xt_s)
    else:
        pz_test = sum(pz_test_parts) / len(pz_test_parts)
    m_ok = mask & np.isfinite(pz_oof) & np.isfinite(oof)
    if not m_ok.any():
        return None
    grid = np.unique(np.concatenate([
        np.array([1e-4, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]),
        np.quantile(pz_oof[m_ok], np.linspace(0.01, 0.99, 60)),
    ]))
    best_thr, best_s = None, base_s
    y_m, o_m, p_m = y[m_ok], oof[m_ok], pz_oof[m_ok]
    min_pos_oof = min(len(y_m), max(2, int(0.02 * len(y_m))))
    min_pos_test = min(len(test), max(2, int(0.01 * len(test))))
    for thr in grid:
        if int((p_m < thr).sum()) < min_pos_oof or int((pz_test < thr).sum()) < min_pos_test:
            continue
        cand = np.where(p_m >= thr, 0.0, o_m)
        s = score("mape", "regression", y_m, cand)
        if np.isfinite(s) and s < best_s - 1e-9:
            best_s, best_thr = s, float(thr)
    if best_thr is None:
        return None
    new_oof = np.where(pz_oof >= best_thr, 0.0, oof)
    new_test = np.where(pz_test >= best_thr, 0.0, test)
    if float(np.ptp(new_test)) <= 1e-8:
        return None
    log(f"zero-MAPE hurdle: порог P(y=0)>={best_thr:.4f} улучшил OOF MAPE {base_s:.4g} -> {best_s:.4g}")
    return {"oof": new_oof, "test": new_test, "score": best_s, "threshold": best_thr}


def finalize_classification(spec, y, mask, oof, test, class_weight_power: float, prior_shift: float = 0.0,
                            weight_power: float = 1.0, shift: LabelShift | None = None):
    """Решающее правило: веса классов под метрику по OOF, сжатые степенью weight_power.

    shift (LabelShift) — оценённый сдвиг долей классов по группам строк: вероятности пересчитываются под
    доли теста, правило подбирается по OOF с весами строк, а его оценка — оценка метрики на тесте.
    Глобальная оценка сдвига частот (EM) только записывается в журнал.
    """
    n_classes = oof.shape[1]
    sw = None
    test_in = test
    if shift is not None and shift.active and spec.metric not in PROBA_METRICS:
        oof, test, sw_all = shift.apply(oof, test)
        sw = sw_all[mask]
    decision = optimize_decision(spec.metric, y[mask], oof[mask], sample_weight=sw)
    if "weights" in decision and weight_power != 1.0:
        decision["weights"] = (np.asarray(decision["weights"], float) ** weight_power).tolist()
        decision["score"] = label_score(spec.metric, y[mask], decide(oof[mask], spec.metric, decision), n_classes, sw)
        decision["weight_power"] = weight_power
    info = {"decision": {k: v for k, v in decision.items() if k != "weights"}}
    if "weights" in decision:
        info["class_weights"] = [round(x, 3) for x in decision["weights"]]
    if sw is not None:
        info["label_shift"] = shift.info
        info["test_label_counts"] = np.bincount(decide(test, spec.metric, decision), minlength=n_classes).tolist()
    test_adj = test
    test = test_in
    if spec.metric not in PROBA_METRICS:
        cnt = np.bincount(y, minlength=n_classes).astype(float)
        eff = cnt * (cnt.max() / np.maximum(cnt, 1)) ** class_weight_power if class_weight_power else cnt
        train_pri = eff / eff.sum()
        mean_pred = test.mean(0)
        est = em_test_priors(test, train_pri)
        tv = 0.5 * np.abs(est - train_pri).sum()
        info["prior_shift_tv"] = round(float(tv), 4)
        info["test_priors_est"] = [round(float(x), 4) for x in est]
        info["train_priors"] = [round(float(x), 4) for x in train_pri]
        info["mean_test_proba"] = [round(float(x), 4) for x in mean_pred]
        if prior_shift and tv > 0.03:
            ratio = (est / np.maximum(train_pri, 1e-9)) ** prior_shift
            test_adj = test * ratio[None, :]
            test_adj = test_adj / test_adj.sum(1, keepdims=True)
            info["prior_shift_applied"] = prior_shift
            log(f"сдвиг классов в тесте (TV={tv:.3f}): поправка вероятностей со степенью {prior_shift}")
    return decision, test_adj, info
