"""Strictly causal expanding-window 1-step-ahead nowcaster for completed-period temporal panels.

When an event table joins regular temporal panels (e.g., hourly telemetry) under a completed-period
rule (t_obs <= floor(t_event, cadence) - cadence), the current period state is unobserved directly
at dispatch time, while the panel itself often has 10x-50x more rows than the labelled events.

This module learns strictly causal 1-step-ahead forecasts (from past lags <= slot - cadence to slot)
using expanding temporal blocks whose boundaries align with validation folds and test start, so:
1. No future or current-period panel observation ever leaks into features, schema, or parameters.
2. Train, validation, and test rows receive identically distributed out-of-sample nowcast features.
"""
import time

import numpy as np
import pandas as pd

from ag_utils import log


def _norm_key_series(s: pd.Series) -> pd.Series:
    """Canonical string normalization for join keys (matches integer/float-int/string consistently)."""
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64").astype("string")
    if pd.api.types.is_float_dtype(s):
        ok = s.dropna()
        if len(ok) and np.all(np.isfinite(ok)) and np.all(ok == np.floor(ok)):
            return s.round().astype("Int64").astype("string")
    return s.astype("string").str.strip()


def _recent_chrono_pos(tr_pos_all: np.ndarray, h_ns: np.ndarray, limit: int) -> np.ndarray:
    """Selects up to `limit` most recent rows chronologically across ALL groups (never slices by group)."""
    if len(tr_pos_all) <= limit:
        return tr_pos_all
    order = np.argsort(h_ns[tr_pos_all], kind="mergesort")
    keep = tr_pos_all[order[-limit:]]
    keep.sort()
    return keep


def _merge_panel_group(panel_list, groups, cadence, early_cut=None):
    """Объединяет несколько панелей с одинаковыми (groups, cadence) в единую мультивариатную сетку."""
    slot_col = "__ag_slot"
    merged = None
    num_cols = []
    for p in panel_list:
        df = p["df"].copy()
        pfx = f"{p['name']}__"
        ren = {c: (c if str(c).startswith(pfx) else f"{pfx}{c}") for c in p["num_cols"]}
        df = df[[*groups, p["slot_col"], *p["num_cols"]]].rename(columns={p["slot_col"]: slot_col, **ren})
        df[slot_col] = pd.to_datetime(df[slot_col], errors="coerce", utc=True)
        for g in groups:
            df[g] = _norm_key_series(df[g])
        # Schema-level non-constancy check strictly on early train history (immune to future poisoning)
        if early_cut is not None:
            early_m = df[slot_col] < early_cut
            if int(early_m.sum()) < 50:
                early_m = df[slot_col] <= df[slot_col].median()
        else:
            early_m = df[slot_col] <= df[slot_col].median()
        valid_c = [c for c in ren.values() if df.loc[early_m, c].nunique(dropna=True) > 1]
        if not valid_c:
            continue
        df = df[[*groups, slot_col, *valid_c]]
        num_cols.extend(valid_c)
        merged = df if merged is None else merged.merge(df, on=[*groups, slot_col], how="outer")
    if merged is None or not num_cols:
        return None, []
    merged = merged.sort_values([*groups, slot_col] if groups else [slot_col]).reset_index(drop=True)
    num_cols = list(dict.fromkeys(num_cols))
    return merged, num_cols


def _build_causal_design(panel: pd.DataFrame, groups: list, slot_col: str, num_cols: list,
                         cadence: pd.Timedelta, early_cut=None):
    """Строит матрицу строго прошлых признаков Z(h) для прогноза состояния панели в слот h."""
    cols = num_cols[:55]
    is_hourly = cadence == pd.Timedelta(hours=1)
    if early_cut is not None:
        early_m = (panel[slot_col] < early_cut).to_numpy()
        if int(early_m.sum()) < 50:
            early_m = (panel[slot_col] <= panel[slot_col].median()).to_numpy()
    else:
        early_m = (panel[slot_col] <= panel[slot_col].median()).to_numpy()

    early_slots = panel.loc[early_m, slot_col]
    span_hours = float((early_slots.max() - early_slots.min()).total_seconds() / 3600.0) if len(early_slots) >= 2 else 0.0

    if groups:
        grp = panel.groupby(groups, sort=False)
        lag1 = grp[cols].shift(1).astype("float32")
        lag2 = grp[cols].shift(2).astype("float32")
        ok1 = (grp[slot_col].diff() == cadence).values
        ok2 = (grp[slot_col].diff(2) == 2 * cadence).values
    else:
        lag1 = panel[cols].shift(1).astype("float32")
        lag2 = panel[cols].shift(2).astype("float32")
        ok1 = (panel[slot_col].diff() == cadence).values
        ok2 = (panel[slot_col].diff(2) == 2 * cadence).values

    lag1.loc[~ok1] = np.nan
    lag2.loc[~ok2] = np.nan

    parts = [
        lag1.add_suffix("__l1"),
        (lag1 - lag2).astype("float32").add_suffix("__d1"),
    ]
    if is_hourly and span_hours >= 72:
        if groups:
            lag24 = grp[cols].shift(24).astype("float32")
            lag25 = grp[cols].shift(25).astype("float32")
            ok24 = (grp[slot_col].diff(24) == pd.Timedelta(hours=24)).values
            ok25 = (grp[slot_col].diff(25) == pd.Timedelta(hours=25)).values
        else:
            lag24 = panel[cols].shift(24).astype("float32")
            lag25 = panel[cols].shift(25).astype("float32")
            ok24 = (panel[slot_col].diff(24) == pd.Timedelta(hours=24)).values
            ok25 = (panel[slot_col].diff(25) == pd.Timedelta(hours=25)).values
        lag24.loc[~ok24] = np.nan
        lag25.loc[~ok25] = np.nan
        parts.append(lag24.add_suffix("__l24"))
        parts.append((lag24 - lag25).astype("float32").add_suffix("__step24"))
    if is_hourly and span_hours >= 24 * 14:
        if groups:
            lag168 = grp[cols].shift(168).astype("float32")
            lag169 = grp[cols].shift(169).astype("float32")
            ok168 = (grp[slot_col].diff(168) == pd.Timedelta(hours=168)).values
            ok169 = (grp[slot_col].diff(169) == pd.Timedelta(hours=169)).values
        else:
            lag168 = panel[cols].shift(168).astype("float32")
            lag169 = panel[cols].shift(169).astype("float32")
            ok168 = (panel[slot_col].diff(168) == pd.Timedelta(hours=168)).values
            ok169 = (panel[slot_col].diff(169) == pd.Timedelta(hours=169)).values
        lag168.loc[~ok168] = np.nan
        lag169.loc[~ok169] = np.nan
        parts.append(lag168.add_suffix("__l168"))
        parts.append((lag168 - lag169).astype("float32").add_suffix("__step168"))

    # Строго каузальная длительность активного бинарного флага на слоте h - cadence (для hazard/survival прогноза)
    rl_dict = {}
    if groups:
        grp_chg = panel[groups[0]].ne(panel[groups[0]].shift(1))
        for gcol in groups[1:]:
            grp_chg = grp_chg | panel[gcol].ne(panel[gcol].shift(1))
        gap_break = grp_chg | (~ok1)
    else:
        gap_break = pd.Series(~ok1, index=panel.index)
    for c in cols:
        head = panel.loc[early_m, c].dropna()
        if len(head) >= 10 and head.nunique() == 2 and head.isin([0, 1, 0.0, 1.0]).mean() >= 0.99:
            v_bin = (panel[c] == 1).astype(np.int8)
            reset = (v_bin == 0) | gap_break
            rl_cur = v_bin.groupby(reset.cumsum(), sort=False).cumsum().astype("float32")
            rl_prev = (rl_cur.groupby([panel[g] for g in groups], sort=False).shift(1)
                       if groups else rl_cur.shift(1)).astype("float32")
            rl_prev.loc[~ok1] = np.nan
            rl_dict[f"{c}__rl1"] = rl_prev
    if rl_dict:
        parts.append(pd.DataFrame(rl_dict, index=panel.index, dtype="float32"))

    # Общесистемное (межгрупповое) среднее по ВСЕМ колонкам на прошлом завершённом слоте h - cadence
    if groups:
        n_groups = int(panel.loc[early_m, groups].drop_duplicates().shape[0])
        if 2 <= n_groups <= 200:
            sys_cols = cols
            sys_by_h = panel.groupby(slot_col, sort=False)[sys_cols].mean().astype("float32")
            prev_h1 = panel[slot_col] - cadence
            sys_l1 = sys_by_h.reindex(prev_h1).reset_index(drop=True)
            sys_l1.columns = [f"sys__{c}" for c in sys_cols]
            parts.append(sys_l1)

    # Календарные признаки целевого слота h (известны в момент события)
    h_dt = panel[slot_col]
    hr = h_dt.dt.hour
    dw = h_dt.dt.dayofweek
    mn = h_dt.dt.month
    doy = h_dt.dt.dayofyear
    cal = pd.DataFrame({
        "__sin_h1": np.sin(2.0 * np.pi * hr / 24.0),
        "__cos_h1": np.cos(2.0 * np.pi * hr / 24.0),
        "__sin_h2": np.sin(4.0 * np.pi * hr / 24.0),
        "__cos_h2": np.cos(4.0 * np.pi * hr / 24.0),
        "__sin_dw": np.sin(2.0 * np.pi * dw / 7.0),
        "__cos_dw": np.cos(2.0 * np.pi * dw / 7.0),
        "__sin_doy": np.sin(2.0 * np.pi * doy / 365.25),
        "__cos_doy": np.cos(2.0 * np.pi * doy / 365.25),
        "__is_wknd": (dw >= 5).astype(np.float32),
        "__month": mn.astype(np.float32),
    }, index=panel.index).astype("float32")
    parts.append(cal)

    if is_hourly:
        parts.append(pd.get_dummies(pd.Categorical(hr, categories=range(24)), prefix="__hr", dtype=np.float32))
    if groups and len(groups) == 1:
        known_g = sorted(panel.loc[early_m, groups[0]].dropna().astype(str).unique())
        if 2 <= len(known_g) <= 60:
            g_cat = pd.Categorical(panel[groups[0]].astype(str), categories=known_g)
            parts.append(pd.get_dummies(g_cat, prefix="__grp", dtype=np.float32))

    Z = pd.concat(parts, axis=1).astype("float32")
    return Z, cols, lag1


def _seasonal_profile(panel: pd.DataFrame, groups: list, slot_col: str, cols: list, train_mask: np.ndarray,
                      is_hourly: bool):
    """Вычисляет строго каузальный сезонный профиль mean(group, hour, is_weekend) по строкам train_mask."""
    h_dt = panel[slot_col]
    key_df = pd.DataFrame(index=panel.index)
    for g in groups:
        key_df[g] = panel[g].astype(str)
    if is_hourly:
        key_df["__h__"] = h_dt.dt.hour.astype(np.int16)
    key_df["__w__"] = (h_dt.dt.dayofweek >= 5).astype(np.int8)
    kcols = list(key_df.columns)
    tr_keys = key_df.loc[train_mask]
    tr_vals = panel.loc[train_mask, cols]
    global_mean = tr_vals.mean(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
    prof_tbl = tr_vals.groupby([tr_keys[c] for c in kcols], sort=False).mean()
    idx_all = pd.MultiIndex.from_frame(key_df) if len(kcols) > 1 else pd.Index(key_df[kcols[0]])
    prof_all = prof_tbl.reindex(idx_all).to_numpy(dtype=np.float32)
    nan_m = ~np.isfinite(prof_all)
    if nan_m.any():
        prof_all = np.where(nan_m, global_mean[None, :], prof_all)
    return prof_all


def _select_top_targets(panel: pd.DataFrame, lag1: pd.DataFrame, cols: list, tr_pidx: np.ndarray,
                        y_train: np.ndarray, tr_rows_mask: np.ndarray, k_top: int = 8) -> list:
    """Выбирает топ-K числовых колонок панели по |corr(lag1, y)| строго на обучающей части."""
    valid = tr_rows_mask & (tr_pidx >= 0) & np.isfinite(y_train)
    if valid.sum() < 100:
        return []
    yv = y_train[valid].astype(float)
    if np.std(yv) < 1e-12:
        return []
    pidx = tr_pidx[valid]
    scored = []
    for c in cols:
        xv = lag1[c].to_numpy(dtype=np.float64)[pidx]
        ok = np.isfinite(xv)
        if ok.sum() < 100 or np.std(xv[ok]) < 1e-12:
            continue
        r = abs(float(np.corrcoef(xv[ok], yv[ok])[0, 1]))
        if np.isfinite(r) and r >= 0.08:
            scored.append((r, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:k_top]]


def build_nowcast_features(rel_meta: dict, y_train: np.ndarray, plan, task_type: str,
                           n_jobs: int = 4, use_lgb_top: bool = True, max_seconds: float = 120.0):
    """Строит строго каузальные out-of-sample nowcast-признаки для train и test.

    Возвращает (df_train_nc, df_test_nc) или (None, None), если подходящих панелей нет.
    """
    if not rel_meta or not rel_meta.get("panels"):
        return None, None
    tr_times_raw = rel_meta.get("train_times")
    te_times_raw = rel_meta.get("test_times")
    tr_groups = rel_meta.get("train_groups")
    te_groups = rel_meta.get("test_groups")
    if tr_times_raw is None or te_times_raw is None or plan.time_col is None:
        return None, None

    tr_times = pd.to_datetime(tr_times_raw, errors="coerce", utc=True).reset_index(drop=True)
    te_times = pd.to_datetime(te_times_raw, errors="coerce", utc=True).reset_index(drop=True)

    t0 = time.time()
    n_tr, n_te = len(tr_times), len(te_times)
    grouped_panels = {}
    for p in rel_meta["panels"]:
        key = (tuple(p["groups"]), p["cadence"])
        grouped_panels.setdefault(key, []).append(p)

    out_tr_parts, out_te_parts = [], []
    y_num = np.asarray(y_train, dtype=float)

    for (grp_tuple, cadence), plist in grouped_panels.items():
        if time.time() - t0 > max_seconds:
            break
        groups = list(grp_tuple)
        if groups and (tr_groups is None or te_groups is None or not set(groups) <= set(tr_groups.columns)):
            continue

        try:
            tr_slots = tr_times.dt.floor(cadence)
            te_slots = te_times.dt.floor(cadence)
        except Exception:
            tr_slots = tr_times.dt.floor("h")
            te_slots = te_times.dt.floor("h")

        fold_starts = sorted({
            tr_slots.iloc[va].min()
            for _, va in plan.folds
            if len(va) and tr_slots.iloc[va].notna().any()
        })
        earliest_val = fold_starts[0] if fold_starts else tr_slots.max()

        panel, num_cols = _merge_panel_group(plist, groups, cadence, early_cut=earliest_val)
        if panel is None or len(panel) < max(4000, 3 * n_tr) or len(panel) > 2_000_000:
            continue

        slot_col = "__ag_slot"
        is_hourly = cadence == pd.Timedelta(hours=1)
        Z_df, cols, lag1 = _build_causal_design(panel, groups, slot_col, num_cols, cadence, early_cut=earliest_val)
        Z_vals = Z_df.to_numpy(dtype=np.float32)
        Y_vals = panel[cols].to_numpy(dtype=np.float32)
        h_series = panel[slot_col]
        h_ns = h_series.astype("int64").to_numpy()

        if groups:
            p_index = pd.MultiIndex.from_frame(panel[[*groups, slot_col]])
            tr_key_df = pd.DataFrame({g: _norm_key_series(tr_groups[g].reset_index(drop=True)) for g in groups})
            tr_key_df[slot_col] = tr_slots.reset_index(drop=True)
            te_key_df = pd.DataFrame({g: _norm_key_series(te_groups[g].reset_index(drop=True)) for g in groups})
            te_key_df[slot_col] = te_slots.reset_index(drop=True)
            tr_pidx = p_index.get_indexer(pd.MultiIndex.from_frame(tr_key_df))
            te_pidx = p_index.get_indexer(pd.MultiIndex.from_frame(te_key_df))
        else:
            p_index = pd.Index(h_series)
            tr_pidx = p_index.get_indexer(tr_slots)
            te_pidx = p_index.get_indexer(te_slots)

        if (tr_pidx >= 0).mean() < 0.5 or not (te_pidx >= 0).any():
            continue

        tr_min_h = tr_slots.min()
        te_min_h = te_slots.min()
        panel_burn = h_series.min() + (pd.Timedelta(hours=168) if is_hourly else 7 * cadence)
        pre_tr_slots = tr_slots[tr_slots < earliest_val].dropna()
        inner_cuts = []
        if pd.notna(tr_min_h) and int(((h_series < tr_min_h) & (h_series >= panel_burn)).sum()) >= 1000:
            inner_cuts.append(tr_min_h)
        if len(pre_tr_slots) >= 200:
            min_cut = max(tr_min_h, panel_burn) + 7 * cadence
            for q_frac in (0.15, 0.55):
                c_ts = pd.Timestamp(pre_tr_slots.quantile(q_frac))
                if c_ts > min_cut and c_ts < earliest_val:
                    if int(((h_series < c_ts) & (h_series >= panel_burn)).sum()) >= 1000:
                        inner_cuts.append(c_ts)
        cuts = sorted(set(inner_cuts + fold_starts))
        test_cutoff = te_min_h if pd.notna(te_min_h) else (tr_slots.max() + cadence)

        n_c = len(cols)
        # Detect binary 0/1 flags strictly on early train history (for calibrated logloss LGBM nowcaster)
        early_p_m = (h_series < earliest_val).to_numpy()
        if int(early_p_m.sum()) < 50:
            early_p_m = (h_series <= h_series.median()).to_numpy()
        is_bin_col = np.zeros(n_c, dtype=bool)
        for j, c in enumerate(cols):
            hd = panel.loc[early_p_m, c].dropna()
            if len(hd) >= 10 and hd.nunique() == 2 and hd.isin([0, 1, 0.0, 1.0]).mean() >= 0.99:
                is_bin_col[j] = True
        tr_pred_r = np.full((n_tr, n_c), np.nan, dtype=np.float32)
        te_pred_r = np.full((n_te, n_c), np.nan, dtype=np.float32)
        tr_prof_out = np.full((n_tr, n_c), np.nan, dtype=np.float32)
        te_prof_out = np.full((n_te, n_c), np.nan, dtype=np.float32)

        pre_val_mask = (tr_slots < earliest_val).to_numpy()
        k_lgb = 6 if max_seconds >= 75.0 else 4
        top_cols = (_select_top_targets(panel, lag1, cols, tr_pidx, y_num, pre_val_mask, k_top=k_lgb)
                    if (use_lgb_top and max_seconds >= 60.0) else [])
        top_idx = [cols.index(c) for c in top_cols]
        tr_pred_g = np.full((n_tr, len(top_cols)), np.nan, dtype=np.float32) if top_cols else None
        te_pred_g = np.full((n_te, len(top_cols)), np.nan, dtype=np.float32) if top_cols else None
        lgb_state = {"enabled": bool(top_cols and (time.time() - t0 < max_seconds * 0.85)), "all_ok": True}

        def _fit_and_predict_block(cutoff_ts, target_rows, target_pidx, out_r, out_prof, out_g, is_test=False):
            p_tr_mask = (h_series < cutoff_ts).to_numpy() & (h_series >= panel_burn).to_numpy()
            if p_tr_mask.sum() < 500:
                return
            ok_rows = np.flatnonzero(target_rows & (target_pidx >= 0))
            if len(ok_rows) == 0:
                return
            ev_pidx = target_pidx[ok_rows]

            # 1) Сезонный профиль (group, hour, is_weekend) строго по p_tr_mask
            prof_all = _seasonal_profile(panel, groups, slot_col, cols, p_tr_mask, is_hourly)
            out_prof[ok_rows] = prof_all[ev_pidx]

            # 2) Многовыходный Cholesky Ridge по остаткам (Y - profile) на недавних хронологических слотах ВСЕХ групп
            tr_pos_all = np.flatnonzero(p_tr_mask)
            tr_pos_r = _recent_chrono_pos(tr_pos_all, h_ns, 150_000)
            Z_tr = Z_vals[tr_pos_r]
            Y_raw_tr = Y_vals[tr_pos_r]
            # Strictly causal domain bounds from past train rows (eliminates impossible negative rainfall/probs)
            with np.errstate(all="ignore"):
                y_lo_tr = np.nanmin(Y_raw_tr, axis=0)
                y_hi_tr = np.nanmax(Y_raw_tr, axis=0)
            span_tr = np.where(np.isfinite(y_hi_tr - y_lo_tr), np.maximum(y_hi_tr - y_lo_tr, 1e-3), 1.0)
            clip_lo = np.where(is_bin_col, 0.0, np.where(y_lo_tr >= 0.0, 0.0, y_lo_tr - 0.15 * span_tr)).astype(np.float32)
            clip_hi = np.where(is_bin_col, 1.0, y_hi_tr + 0.15 * span_tr).astype(np.float32)

            Y_tr = Y_raw_tr - prof_all[tr_pos_r]
            col_fill = np.nanmean(Z_tr, axis=0)
            col_fill = np.where(np.isfinite(col_fill), col_fill, 0.0).astype(np.float32)
            Z_tr_f = np.where(np.isfinite(Z_tr), Z_tr, col_fill[None, :])
            mu = Z_tr_f.mean(axis=0, dtype=np.float64).astype(np.float32)
            sd = Z_tr_f.std(axis=0, dtype=np.float64).astype(np.float32)
            sd = np.where(sd > 1e-6, sd, 1.0)
            Zs = (Z_tr_f - mu) / sd
            Y_f = np.where(np.isfinite(Y_tr), Y_tr, 0.0).astype(np.float32)
            y_bias = Y_f.mean(axis=0, dtype=np.float64).astype(np.float32)
            Yc = Y_f - y_bias

            try:
                from threadpoolctl import threadpool_limits
                _lim_ctx = threadpool_limits(limits=min(4, max(1, n_jobs)))
            except Exception:
                from contextlib import nullcontext
                _lim_ctx = nullcontext()
            with _lim_ctx:
                G = (Zs.T @ Zs).astype(np.float64)
                G[np.diag_indices_from(G)] += 300.0
                R = (Zs.T @ Yc).astype(np.float64)
                try:
                    W = np.linalg.solve(G, R)
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(G, R, rcond=None)[0]

                # float64 prediction matmul guarantees bit-exact float32 invariance across test batch sizes
                Z_ev = Z_vals[ev_pidx]
                Z_ev_f = np.where(np.isfinite(Z_ev), Z_ev, col_fill[None, :])
                Z_ev_s = ((Z_ev_f - mu) / sd).astype(np.float64)
                pred_res = (Z_ev_s @ W + y_bias.astype(np.float64)).astype(np.float32)
            raw_pred_r = prof_all[ev_pidx] + pred_res
            out_r[ok_rows] = np.clip(raw_pred_r, clip_lo[None, :], clip_hi[None, :]).astype(np.float32)

            # 3) Нелинейный LightGBM nowcaster: binary logloss для 0/1 флагов и L2-остатки для непрерывных сенсоров
            if lgb_state["enabled"] and out_g is not None:
                if not is_test and (time.time() - t0 > max_seconds * 0.95):
                    lgb_state["enabled"] = False
                    lgb_state["all_ok"] = False
                    return
                import lightgbm as lgb

                tr_pos = _recent_chrono_pos(tr_pos_all, h_ns, 40_000)
                Z_g_tr = Z_vals[tr_pos]
                Z_g_ev = Z_vals[ev_pidx]
                params_g = {
                    "objective": "regression",
                    "learning_rate": 0.1,
                    "num_leaves": 31,
                    "max_bin": 63,
                    "min_child_samples": 60,
                    "feature_fraction": 0.65,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "lambda_l2": 5.0,
                    "verbose": -1,
                    "num_threads": max(1, min(8, n_jobs)),
                    "seed": 42,
                }
                for j, ci in enumerate(top_idx):
                    if is_bin_col[ci]:
                        yb = Y_vals[tr_pos, ci]
                        ok_y = np.isfinite(yb)
                        if ok_y.sum() < 500:
                            continue
                        yb_ok = (yb[ok_y] >= 0.5).astype(np.float32)
                        if yb_ok.min() == yb_ok.max():
                            out_g[ok_rows, j] = float(yb_ok[0])
                            continue
                        ds = lgb.Dataset(Z_g_tr[ok_y], yb_ok, free_raw_data=True)
                        m_g = lgb.train({**params_g, "objective": "binary"}, ds, num_boost_round=55)
                        out_g[ok_rows, j] = np.clip(m_g.predict(Z_g_ev), 0.0, 1.0).astype(np.float32)
                    else:
                        yj = Y_vals[tr_pos, ci] - prof_all[tr_pos, ci]
                        ok_y = np.isfinite(yj)
                        if ok_y.sum() < 500:
                            continue
                        ds = lgb.Dataset(Z_g_tr[ok_y], yj[ok_y], free_raw_data=True)
                        m_g = lgb.train(params_g, ds, num_boost_round=55)
                        pg = prof_all[ev_pidx, ci] + m_g.predict(Z_g_ev).astype(np.float32)
                        out_g[ok_rows, j] = np.clip(pg, clip_lo[ci], clip_hi[ci]).astype(np.float32)

        for bi, start_ts in enumerate(cuts):
            end_ts = cuts[bi + 1] if bi + 1 < len(cuts) else None
            if end_ts is not None:
                blk_mask = (tr_slots >= start_ts).to_numpy() & (tr_slots < end_ts).to_numpy()
            else:
                blk_mask = (tr_slots >= start_ts).to_numpy()
            if blk_mask.any():
                _fit_and_predict_block(start_ts, blk_mask, tr_pidx, tr_pred_r, tr_prof_out, tr_pred_g, is_test=False)

        _fit_and_predict_block(test_cutoff, np.ones(n_te, dtype=bool), te_pidx, te_pred_r, te_prof_out, te_pred_g,
                               is_test=True)

        l1_tr = lag1[cols].to_numpy(dtype=np.float32)
        tr_l1_ev = np.where((tr_pidx >= 0)[:, None], l1_tr[np.maximum(tr_pidx, 0)], np.nan)
        te_l1_ev = np.where((te_pidx >= 0)[:, None], l1_tr[np.maximum(te_pidx, 0)], np.nan)

        tr_dict, te_dict = {}, {}
        for j, c in enumerate(cols):
            tr_dict[f"nc__{c}"] = tr_pred_r[:, j]
            te_dict[f"nc__{c}"] = te_pred_r[:, j]
            tr_dict[f"nc_d1__{c}"] = (tr_pred_r[:, j] - tr_l1_ev[:, j]).astype("float32")
            te_dict[f"nc_d1__{c}"] = (te_pred_r[:, j] - te_l1_ev[:, j]).astype("float32")
            tr_dict[f"nc_prof__{c}"] = tr_prof_out[:, j]
            te_dict[f"nc_prof__{c}"] = te_prof_out[:, j]
        use_gb_out = bool(top_cols and tr_pred_g is not None and lgb_state["enabled"] and lgb_state["all_ok"])
        if use_gb_out:
            for j, c in enumerate(top_cols):
                tr_dict[f"nc_gb__{c}"] = tr_pred_g[:, j]
                te_dict[f"nc_gb__{c}"] = te_pred_g[:, j]
                tr_dict[f"nc_gbd__{c}"] = (tr_pred_g[:, j] - tr_pred_r[:, top_idx[j]]).astype("float32")
                te_dict[f"nc_gbd__{c}"] = (te_pred_g[:, j] - te_pred_r[:, top_idx[j]]).astype("float32")

        out_tr_parts.append(pd.DataFrame(tr_dict, dtype="float32"))
        out_te_parts.append(pd.DataFrame(te_dict, dtype="float32"))
        log(
            f"nowcast: группа={groups}, шаг={cadence}, блоков={len(cuts)}, "
            f"колонок Ridge={len(cols)}, LGBM={len(top_cols) if use_gb_out else 0}, за {time.time() - t0:.1f}s"
        )

    if not out_tr_parts:
        return None, None
    return pd.concat(out_tr_parts, axis=1), pd.concat(out_te_parts, axis=1)
