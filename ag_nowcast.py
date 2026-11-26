"""Strictly causal expanding-window 1-step-ahead nowcaster for completed-period temporal panels.

When an event table joins regular temporal panels (e.g., hourly telemetry) under a completed-period
rule (t_obs <= floor(t_event, cadence) - cadence), the current period state is unobserved directly
at dispatch time, while the panel itself often has 10x-50x more rows than the labelled events.

This module learns strictly causal 1-step-ahead forecasts (from past lags <= slot - cadence to slot)
using expanding temporal blocks whose boundaries align with validation folds and test start, so:
1. No future or current-period panel observation ever leaks into features or model parameters.
2. Train, validation, and test rows receive identically distributed out-of-sample nowcast features.
"""
import time

import numpy as np
import pandas as pd

from ag_utils import log


def _merge_panel_group(panel_list, groups, cadence):
    """Объединяет несколько панелей с одинаковыми (groups, cadence) в единую мультивариатную сетку."""
    slot_col = "__ag_slot"
    merged = None
    num_cols = []
    for p in panel_list:
        df = p["df"].copy()
        pfx = f"{p['name']}__"
        ren = {c: (c if str(c).startswith(pfx) else f"{pfx}{c}") for c in p["num_cols"]}
        df = df[[*groups, p["slot_col"], *p["num_cols"]]].rename(columns={p["slot_col"]: slot_col, **ren})
        valid_c = [c for c in ren.values() if df[c].nunique(dropna=True) > 1]
        df = df[[*groups, slot_col, *valid_c]]
        num_cols.extend(valid_c)
        merged = df if merged is None else merged.merge(df, on=[*groups, slot_col], how="outer")
    if merged is None or not num_cols:
        return None, []
    merged = merged.sort_values([*groups, slot_col] if groups else [slot_col]).reset_index(drop=True)
    num_cols = list(dict.fromkeys(num_cols))
    return merged, num_cols


def _build_causal_design(panel: pd.DataFrame, groups: list, slot_col: str, num_cols: list, cadence: pd.Timedelta):
    """Строит матрицу строго прошлых признаков Z(h) для прогноза состояния панели в слот h."""
    cols = num_cols[:55]
    is_hourly = cadence == pd.Timedelta(hours=1)
    span_hours = (panel[slot_col].max() - panel[slot_col].min()).total_seconds() / 3600.0

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

    # Общесистемное (межгрупповое) среднее на предыдущем завершённом слоте h - cadence
    if groups:
        n_groups = int(panel[groups].drop_duplicates().shape[0])
        if 2 <= n_groups <= 200:
            sys_cols = cols[:25]
            sys_by_h = panel.groupby(slot_col, sort=False)[sys_cols].mean().astype("float32")
            prev_h = panel[slot_col] - cadence
            sys_l1 = sys_by_h.reindex(prev_h).reset_index(drop=True)
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
        parts.append(pd.get_dummies(hr, prefix="__hr", dtype=np.float32))
    if groups and len(groups) == 1:
        n_g = panel[groups[0]].nunique(dropna=True)
        if 2 <= n_g <= 60:
            parts.append(pd.get_dummies(panel[groups[0]].astype(str), prefix="__grp", dtype=np.float32))

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
                        y_train: np.ndarray, tr_rows_mask: np.ndarray, k_top: int = 6) -> list:
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
    tr_times = rel_meta.get("train_times")
    te_times = rel_meta.get("test_times")
    tr_groups = rel_meta.get("train_groups")
    te_groups = rel_meta.get("test_groups")
    if tr_times is None or te_times is None or plan.time_col is None:
        return None, None

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
        panel, num_cols = _merge_panel_group(plist, groups, cadence)
        if panel is None or len(panel) < max(4000, 3 * n_tr) or len(panel) > 2_000_000:
            continue
        if groups and (tr_groups is None or not set(groups) <= set(tr_groups.columns)):
            continue

        slot_col = "__ag_slot"
        is_hourly = cadence == pd.Timedelta(hours=1)
        Z_df, cols, lag1 = _build_causal_design(panel, groups, slot_col, num_cols, cadence)
        Z_vals = Z_df.to_numpy(dtype=np.float32)
        Y_vals = panel[cols].to_numpy(dtype=np.float32)
        h_series = panel[slot_col]

        try:
            tr_slots = tr_times.dt.floor(cadence)
            te_slots = te_times.dt.floor(cadence)
        except Exception:
            tr_slots = tr_times.dt.floor("h")
            te_slots = te_times.dt.floor("h")

        if groups:
            p_index = pd.MultiIndex.from_frame(panel[[*groups, slot_col]])
            tr_key_df = tr_groups[groups].reset_index(drop=True).copy()
            tr_key_df[slot_col] = tr_slots.reset_index(drop=True)
            te_key_df = te_groups[groups].reset_index(drop=True).copy()
            te_key_df[slot_col] = te_slots.reset_index(drop=True)
            tr_pidx = p_index.get_indexer(pd.MultiIndex.from_frame(tr_key_df))
            te_pidx = p_index.get_indexer(pd.MultiIndex.from_frame(te_key_df))
        else:
            p_index = pd.Index(h_series)
            tr_pidx = p_index.get_indexer(tr_slots)
            te_pidx = p_index.get_indexer(te_slots)

        if (tr_pidx >= 0).mean() < 0.5 or not (te_pidx >= 0).any():
            continue

        fold_starts = sorted({
            tr_slots.iloc[va].min()
            for _, va in plan.folds
            if len(va) and tr_slots.iloc[va].notna().any()
        })
        earliest_val = fold_starts[0] if fold_starts else tr_slots.max()
        tr_min_h = tr_slots.min()
        te_min_h = te_slots.min()
        pre_tr_slots = tr_slots[tr_slots < earliest_val].dropna()
        inner_cuts = []
        if len(pre_tr_slots) >= 200:
            for q_frac in (0.25, 0.60):
                c_ts = pd.Timestamp(pre_tr_slots.quantile(q_frac))
                if c_ts > tr_min_h + 7 * cadence and c_ts < earliest_val:
                    inner_cuts.append(c_ts)
        cuts = sorted(set(inner_cuts + fold_starts))
        test_cutoff = te_min_h if pd.notna(te_min_h) else (tr_slots.max() + cadence)

        n_c = len(cols)
        tr_pred_r = np.full((n_tr, n_c), np.nan, dtype=np.float32)
        te_pred_r = np.full((n_te, n_c), np.nan, dtype=np.float32)
        tr_prof_out = np.full((n_tr, n_c), np.nan, dtype=np.float32)
        te_prof_out = np.full((n_te, n_c), np.nan, dtype=np.float32)

        pre_val_mask = (tr_slots < earliest_val).to_numpy()
        k_lgb = 6 if max_seconds >= 90.0 else 4
        top_cols = _select_top_targets(panel, lag1, cols, tr_pidx, y_num, pre_val_mask, k_top=k_lgb) if (use_lgb_top and max_seconds >= 60.0) else []
        top_idx = [cols.index(c) for c in top_cols]
        tr_pred_g = np.full((n_tr, len(top_cols)), np.nan, dtype=np.float32) if top_cols else None
        te_pred_g = np.full((n_te, len(top_cols)), np.nan, dtype=np.float32) if top_cols else None
        lgb_state = {"enabled": bool(top_cols and (time.time() - t0 < max_seconds * 0.85)), "all_ok": True}

        panel_burn = h_series.min() + (pd.Timedelta(hours=168) if is_hourly else 7 * cadence)

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

            # 2) Быстрый многовыходный Cholesky Ridge по остаткам (Y - profile) на недавних прошлых слотах
            tr_pos_all = np.flatnonzero(p_tr_mask)
            if len(tr_pos_all) > 150_000:
                tr_pos_r = tr_pos_all[-150_000:]
            else:
                tr_pos_r = tr_pos_all
            Z_tr = Z_vals[tr_pos_r]
            Y_tr = Y_vals[tr_pos_r] - prof_all[tr_pos_r]
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

            # Решаем нормальные уравнения (Zs^T Zs + alpha I) W = Zs^T Yc
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
                    W = np.linalg.solve(G, R).astype(np.float32)
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(G, R, rcond=None)[0].astype(np.float32)

                Z_ev = Z_vals[ev_pidx]
                Z_ev_f = np.where(np.isfinite(Z_ev), Z_ev, col_fill[None, :])
                pred_res = (((Z_ev_f - mu) / sd) @ W + y_bias).astype(np.float32)
            out_r[ok_rows] = prof_all[ev_pidx] + pred_res

            # 3) Нелинейный LightGBM nowcaster для топ-K ключевых сенсорных колонок
            if lgb_state["enabled"] and out_g is not None:
                if not is_test and (time.time() - t0 > max_seconds * 0.95):
                    lgb_state["enabled"] = False
                    lgb_state["all_ok"] = False
                    return
                import lightgbm as lgb

                tr_pos = np.flatnonzero(p_tr_mask)
                if len(tr_pos) > 50_000:
                    tr_pos = tr_pos[-50_000:]
                Z_g_tr = Z_vals[tr_pos]
                Z_g_ev = Z_vals[ev_pidx]
                params_g = {
                    "objective": "regression",
                    "learning_rate": 0.1,
                    "num_leaves": 31,
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
                    yj = Y_vals[tr_pos, ci] - prof_all[tr_pos, ci]
                    ok_y = np.isfinite(yj)
                    if ok_y.sum() < 500:
                        continue
                    ds = lgb.Dataset(Z_g_tr[ok_y], yj[ok_y], free_raw_data=True)
                    m_g = lgb.train(params_g, ds, num_boost_round=60)
                    out_g[ok_rows, j] = prof_all[ev_pidx, ci] + m_g.predict(Z_g_ev).astype(np.float32)

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
