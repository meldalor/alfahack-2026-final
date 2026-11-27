"""Schema-driven joins: unique dimensions, bounded temporal lookup, intervals, and causal one-to-many."""
import itertools
import re
import numpy as np
import pandas as pd
from ag_utils import log, log_exc

TIME_NAMES = {'timestamp', 'date', 'datetime', 'time', 'event_time', 'created_at', 'updated_at', 'recorded_at', 'ts', 'dt'}
POST_FACTUM_RE = re.compile(r'(?:^|_)(?:actual|final|realized|outcome|result|resolved)(?:_|$)', re.I)
_DMY_DOT_RE = re.compile(r'^\s*\d{1,2}\.\d{1,2}\.\d{2,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*$')


def _unix_epoch_unit(series: pd.Series) -> str | None:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) < 3 or s.nunique() < 3:
        return None
    vals = s.to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return None
    lo, hi = float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))
    if 9e8 <= lo and hi <= 2.6e9:
        return 's'
    if 9e11 <= lo and hi <= 2.6e12:
        return 'ms'
    return None


def _parse_time_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors='coerce', utc=True)
    unit = _unix_epoch_unit(series)
    if unit is not None:
        return pd.to_datetime(pd.to_numeric(series, errors='coerce'), unit=unit, errors='coerce', utc=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample = series.dropna().astype(str).head(100)
        dayfirst = bool(len(sample) and sample.map(lambda v: bool(_DMY_DOT_RE.match(v))).mean() > 0.7)
        parsed = pd.to_datetime(series, errors='coerce', utc=True, dayfirst=dayfirst)
        if parsed.notna().mean() < 0.5 and series.notna().any():
            parsed = pd.to_datetime(series, errors='coerce', utc=True, format='mixed', dayfirst=dayfirst)
    return parsed


def _is_time_col(name: str, series: pd.Series | None = None) -> bool:
    low = str(name).lower()
    if low.startswith(('start_', 'end_')):
        return False
    name_ok = (low in TIME_NAMES or low.endswith(('_time', '_timestamp', '_date', '_datetime', '_at', '_ts', '_dt', '_epoch'))
               or low.startswith(('timestamp_', 'datetime_', 'event_time', 'unix_')))
    if not name_ok:
        return False
    if series is None:
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return _unix_epoch_unit(series) is not None
    sample = series.dropna().head(200)
    if len(sample) < 3:
        return False
    parsed = _parse_time_series(sample)
    return float(parsed.notna().mean()) >= 0.9


def _norm_key_series(s: pd.Series) -> pd.Series:
    """Canonical string representation for mismatched key dtypes (e.g., int 10 vs float 10.0 vs str ' 10 ')."""
    if pd.api.types.is_float_dtype(s):
        vals = s.values
        finite = np.isfinite(vals)
        if finite.any() and np.all(np.isclose(vals[finite], np.round(vals[finite]))):
            out = pd.Series(pd.NA, index=s.index, dtype='string')
            out.loc[finite] = pd.Series(np.round(vals[finite]).astype(np.int64), index=s.index[finite]).astype('string')
            return out
    return s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def _align_key_dtypes(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]):
    """Align join key dtypes safely without crashing on int/str/float mismatches."""
    l_out, r_out = left, right
    changed = False
    for k in keys:
        if k not in l_out.columns or k not in r_out.columns:
            continue
        a, b = l_out[k], r_out[k]
        if a.dtype == b.dtype:
            if pd.api.types.is_object_dtype(a) or pd.api.types.is_string_dtype(a):
                sa = a.astype('string').str.strip()
                sb = b.astype('string').str.strip()
                if not sa.equals(a.astype('string')) or not sb.equals(b.astype('string')):
                    if not changed:
                        l_out, r_out, changed = l_out.copy(), r_out.copy(), True
                    l_out[k], r_out[k] = sa, sb
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            continue
        if not changed:
            l_out, r_out, changed = l_out.copy(), r_out.copy(), True
        l_out[k] = _norm_key_series(a)
        r_out[k] = _norm_key_series(b)
    return l_out, r_out


def _keys(dim, common):
    """Prefer truly unique keys, then discard rare ambiguous keys in dirty dimensions."""
    clean = dim.drop_duplicates()
    candidates = [list(c) for size in range(1, min(3, len(common)) + 1)
                  for c in itertools.combinations(common, size)]
    for allow_dirty in (False, True):
        for cols in candidates:
            nonnull = clean[cols].notna().all(axis=1)
            valid = nonnull & ~clean.duplicated(cols, keep=False)
            if valid.all() or (allow_dirty and valid.mean() >= 0.95):
                return cols, clean.loc[valid].copy(), int((~valid).sum())
    return None, None, 0


def _temporal_join(left, right, keys, marker, panels_out=None, table_name=None):
    times = [c for c in keys if _is_time_col(c, right[c])]
    if len(times) != 1:
        return None
    tc = times[0]
    groups = [c for c in keys if c != tc]
    left, right = _align_key_dtypes(left, right, groups)
    rt = _parse_time_series(right[tc])
    lt = _parse_time_series(left[tc])
    if rt.notna().mean() < .99 or lt.notna().mean() < .9:
        return None
    # Require a stable measurement cadence. Never carry stale observations over outages.
    series = right[keys].copy(); series[tc] = rt
    series = series.sort_values(tc)
    diffs = series.groupby(groups, dropna=False)[tc].diff() if groups else series[tc].diff()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if len(diffs) < 3:
        return None
    cadence = diffs.median()
    if cadence <= pd.Timedelta(0) or (diffs == cadence).mean() < .8:
        return None
    key = '__ag_join_time__'; row = '__ag_join_row__'
    if key in left or row in left or key in right or row in right:
        raise ValueError('Reserved relational helper column collision')
    a = left.copy(); a[key] = lt; a[row] = np.arange(len(a))
    b = right.drop(columns=[tc]).copy(); b[key] = rt
    valid = a[key].notna()
    for g in groups:
        valid &= a[g].notna()
    aok = a.loc[valid].sort_values(key)
    b = b.loc[b[key].notna()].sort_values(key)
    # Bounded backward forward-fill for transient sensor outages (strictly past observations, no future leakage).
    base_num = [c for c in b.columns if c not in groups and c not in (key, marker)
                and pd.api.types.is_numeric_dtype(b[c])]
    fill_cols = [c for c in base_num if b[c].isna().any()]
    if fill_cols:
        null_col = f'{marker[:-len("__matched")]}__raw_nulls' if marker.endswith('__matched') else f'{marker}__raw_nulls'
        b[null_col] = b[fill_cols].isna().sum(axis=1).astype('float32')
        if groups:
            b[fill_cols] = b.groupby(groups, sort=False)[fill_cols].ffill(limit=6)
        else:
            b[fill_cols] = b[fill_cols].ffill(limit=6)
    # Export strictly-past regular panel snapshot for causal 1-step-ahead nowcaster (before shifting key by +cadence)
    if isinstance(panels_out, list) and base_num and len(b) >= 4000:
        slot_col = '__ag_slot__'
        pcols = base_num[:16]
        pdf = b[groups + [key] + pcols].copy()
        pdf = pdf.rename(columns={key: slot_col})
        for c in pcols:
            pdf[c] = pdf[c].astype('float32')
        panels_out.append({
            'name': table_name or marker[:-len('__matched')],
            'df': pdf,
            'groups': list(groups),
            'slot_col': slot_col,
            'cadence': cadence,
            'num_cols': list(pcols),
            'time_col': tc,
        })
    # Causal temporal dynamics from strictly completed past observations (all shifts >= 1, strictly <= rt <= ev - cadence).
    n_groups = int(b.groupby(groups, dropna=False).ngroups) if groups else 1
    per_grp = len(b) / max(n_groups, 1)
    if base_num and per_grp >= 12:
        b = b.sort_values((groups + [key]) if groups else [key]).reset_index(drop=True)
        gb = b.groupby(groups, sort=False) if groups else None
        extra = {}
        is_hourly = (cadence == pd.Timedelta(hours=1))
        is_daily = (cadence == pd.Timedelta(days=1))
        add_sys = bool(groups and 2 <= n_groups <= 200)
        early_mask = (b[key] <= b[key].median()).values
        for c in base_num[:16]:
            s = b[c].astype('float32')
            s1 = (gb[c].shift(1) if gb is not None else s.shift(1)).astype('float32')
            extra[f'{c}__d1'] = (s - s1).astype('float32')
            rm3 = (gb[c].transform(lambda x: x.rolling(3, min_periods=1).mean())
                   if gb is not None else s.rolling(3, min_periods=1).mean())
            extra[f'{c}__rm3'] = rm3.astype('float32')
            if add_sys:
                # Cross-group (system-wide) aggregates strictly at the same completed timestamp rt <= ev - cadence.
                sys_mean = b.groupby(key, sort=False)[c].transform('mean').astype('float32')
                extra[f'{c}__sys_mean'] = sys_mean
                extra[f'{c}__sys_dev'] = (s - sys_mean).astype('float32')
            # Strictly causal run-length of active binary state (binary check strictly on first temporal half).
            head = b.loc[early_mask, c].dropna()
            if len(head) >= 10 and head.nunique() == 2 and head.isin([0, 1, 0.0, 1.0]).mean() >= 0.99:
                v_bin = (b[c] == 1).astype(np.int8)
                if groups:
                    grp_chg = b[groups[0]].ne(b[groups[0]].shift(1))
                    for gcol in groups[1:]:
                        grp_chg = grp_chg | b[gcol].ne(b[gcol].shift(1))
                    reset = (v_bin == 0) | grp_chg
                else:
                    reset = (v_bin == 0)
                extra[f'{c}__runlen'] = v_bin.groupby(reset.cumsum(), sort=False).cumsum().astype('float32')
            if is_hourly and per_grp >= 200:
                # Shifts 23,47,...,167 from completed hour t = exact target hour (t+1h) over last 7 completed days!
                d_tgt = [(gb[c].shift(k) if gb is not None else s.shift(k)).astype('float32')
                         for k in (23, 47, 71, 95, 119, 143, 167)]
                h7d = pd.concat(d_tgt, axis=1).mean(axis=1).astype('float32')
                extra[f'{c}__h7d_tgt'] = h7d
                if per_grp >= 700:
                    # Same weekday & exact target hour over last 4 completed weeks (167, 335, 503, 671h before t).
                    w_tgt = [(gb[c].shift(k) if gb is not None else s.shift(k)).astype('float32')
                             for k in (167, 335, 503, 671)]
                    w_cur = [(gb[c].shift(k) if gb is not None else s.shift(k)).astype('float32')
                             for k in (168, 336, 504, 672)]
                    dow4_tgt = pd.concat(w_tgt, axis=1).mean(axis=1).astype('float32')
                    dow4_cur = pd.concat(w_cur, axis=1).mean(axis=1).astype('float32')
                    anom = (s - dow4_cur).astype('float32')
                    extra[f'{c}__dow4_tgt'] = dow4_tgt
                    extra[f'{c}__anom'] = anom
                    extra[f'{c}__fcst'] = (dow4_tgt + 0.65 * anom).astype('float32')
            elif is_daily and per_grp >= 35:
                w_tgt = [(gb[c].shift(k) if gb is not None else s.shift(k)).astype('float32')
                         for k in (6, 13, 20, 27)]
                extra[f'{c}__dow4_tgt'] = pd.concat(w_tgt, axis=1).mean(axis=1).astype('float32')
        if extra:
            b = pd.concat([b, pd.DataFrame(extra, index=b.index)], axis=1)
    # STRICT COMPLETED-INTERVAL AVAILABILITY:
    # A periodic measurement stamped at `rt` covers [rt, rt + cadence) and completes at `rt + cadence`.
    # Matching `rt + cadence <= event_time` (with tolerance < cadence) guarantees we NEVER use an in-progress interval
    # (e.g., for an event at 10:30 with 1h cadence, the 10:00-11:00 record is excluded; last allowed is 09:00-10:00).
    b[key] = b[key] + cadence
    b = b.sort_values(key)
    joined = pd.merge_asof(aok, b, on=key, by=groups or None, direction='backward', allow_exact_matches=True,
                           tolerance=cadence - pd.Timedelta(nanoseconds=1))
    missing = a.loc[~valid]
    joined = pd.concat([joined, missing], ignore_index=True).sort_values(row)
    joined = joined.drop(columns=[key, row]).reset_index(drop=True)
    return joined, cadence


def _interval_join(left, dim, name, targets):
    """Generic interval join for tables with (start_*, end_*) timestamps and shared group keys."""
    l_times = [c for c in left.columns if _is_time_col(c, left[c])]
    if len(l_times) != 1:
        return None
    ltc = l_times[0]
    starts = [c for c in dim.columns if c.lower().startswith('start_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
    ends = [c for c in dim.columns if c.lower().startswith('end_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
    if len(starts) != 1 or len(ends) != 1:
        return None
    sc, ec = starts[0], ends[0]
    groups = []
    for c in dim.columns:
        if c not in left.columns or c in targets or not (c.lower().endswith('_id') or c.lower() == 'id'):
            continue
        if c.lower() == 'event_id' and left[c].nunique() > 0.9 * len(left):
            continue
        ls = _norm_key_vals(left[c])
        ds = _norm_key_vals(dim[c])
        dvals = set(ds.dropna().unique())
        if dvals and float(ls.isin(dvals).mean()) > 0.0:
            groups.append(c)
    if not groups or len(dim) > 50_000:
        return None
    lt = _parse_time_series(left[ltc])
    st = _parse_time_series(dim[sc])
    et = _parse_time_series(dim[ec])
    if lt.notna().mean() < 0.9 or st.notna().mean() < 0.9 or et.notna().mean() < 0.9:
        return None
    a = left[groups].copy(); a['__t__'] = lt; a['__row__'] = np.arange(len(left))
    keep_cols = [c for c in dim.columns if c not in groups and c not in (sc, ec) and c not in targets
                 and not c.lower().endswith('_id') and c.lower() != 'id'
                 and not POST_FACTUM_RE.search(str(c))]
    b = dim[groups + keep_cols].copy(); b['__s__'] = st; b['__e__'] = et
    a, b = _align_key_dtypes(a, b, groups)
    b = b.dropna(subset=['__s__', '__e__'] + groups)
    if b.empty:
        return None
    m = a.dropna(subset=['__t__'] + groups).merge(b, on=groups, how='inner', sort=False)
    m = m[(m['__t__'] >= m['__s__']) & (m['__t__'] <= m['__e__'])]
    if m.empty:
        return None
    out = pd.DataFrame(index=np.arange(len(left)))
    cnt = m.groupby('__row__').size()
    out[f'{name}__active_count'] = out.index.map(cnt).fillna(0).astype('float32')
    for c in keep_cols:
        if pd.api.types.is_numeric_dtype(m[c]):
            agg = m.groupby('__row__')[c].max()
            out[f'{name}__{c}'] = out.index.map(agg).astype('float32')
        elif m[c].nunique(dropna=True) <= 50:
            agg = m.groupby('__row__')[c].first()
            out[f'{name}__{c}'] = out.index.map(agg)
    return out


def _aggregate_child(left, dim, name, targets):
    """Generic one-to-many child table aggregation (causal prefix-sum if timed, static groupby otherwise)."""
    if len(dim) == 0 or len(dim) > 2_000_000:
        return None
    # Never aggregate interval tables (start_*, end_*) via static groupby (prevents future/post-factum leakage).
    _st_c = [c for c in dim.columns if c.lower().startswith('start_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
    _en_c = [c for c in dim.columns if c.lower().startswith('end_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
    if _st_c and _en_c:
        return None
    l_id_cols = [c for c in left.columns if c in dim.columns and c not in targets
                 and (c.lower().endswith('_id') or c.lower() == 'id')
                 and not (c.lower() == 'event_id' and left[c].nunique() > 0.9 * len(left))]
    if not l_id_cols:
        return None
    # Pick the shared ID column with highest non-zero overlap and reasonable cardinality.
    best_key, best_cov = None, 0.0
    for c in l_id_cols:
        ls = _norm_key_series(left[c]) if left[c].dtype != dim[c].dtype else left[c]
        ds = _norm_key_series(dim[c]) if left[c].dtype != dim[c].dtype else dim[c]
        lvals = set(ls.dropna().unique())
        if not lvals:
            continue
        cov = float(ds.isin(lvals).mean())
        if cov > best_cov and cov >= 0.05:
            best_key, best_cov = c, cov
    if best_key is None:
        return None
    left_al, dim_al = _align_key_dtypes(left[[best_key]], dim, [best_key])
    d_times = [c for c in dim_al.columns if _is_time_col(c, dim_al[c])]
    l_times = [c for c in left.columns if _is_time_col(c, left[c])]
    num_cols = [c for c in dim_al.columns if c != best_key and c not in targets and c not in d_times
                and not c.lower().endswith('_id') and c.lower() != 'id'
                and not POST_FACTUM_RE.search(str(c))
                and pd.api.types.is_numeric_dtype(dim_al[c])][:8]
    cat_cols = [c for c in dim_al.columns if c != best_key and c not in targets and c not in d_times
                and c not in num_cols and not c.lower().endswith('_id') and c.lower() != 'id'
                and not POST_FACTUM_RE.search(str(c))
                and 2 <= dim_al[c].nunique(dropna=True) <= 200][:4]
    if d_times:
        if len(d_times) != 1 or len(l_times) != 1:
            return None  # Timed child table without matching event clock: skip to prevent future leakage.
        dtc, ltc = d_times[0], l_times[0]
        dt = _parse_time_series(dim_al[dtc])
        lt = _parse_time_series(left[ltc])
        if dt.notna().mean() < 0.9 or lt.notna().mean() < 0.9:
            return None
        b = dim_al[[best_key] + num_cols].copy()
        b['__t__'] = dt
        b = b.dropna(subset=[best_key, '__t__']).sort_values([best_key, '__t__']).reset_index(drop=True)
        if b.empty:
            return None
        pref = pd.DataFrame({best_key: b[best_key], '__t__': b['__t__']})
        g = b.groupby(best_key, sort=False)
        cnt = g.cumcount() + 1
        pref[f'{name}__hist_count'] = cnt.astype('float32')
        first_t = g['__t__'].transform('first')
        pref['__first_t__'] = first_t
        pref['__last_t__'] = b['__t__']
        for idx_c, c in enumerate(num_cols):
            v = b[c].astype('float64')
            n_ok = v.notna().groupby(b[best_key], sort=False).cumsum()
            csum = v.fillna(0.0).groupby(b[best_key], sort=False).cumsum()
            cmean = csum / n_ok.replace(0, np.nan)
            clast = v.groupby(b[best_key], sort=False).ffill()
            pref[f'{name}__{c}__hist_mean'] = cmean.astype('float32')
            pref[f'{name}__{c}__hist_last'] = clast.astype('float32')
            pref[f'{name}__{c}__hist_max'] = v.groupby(b[best_key], sort=False).cummax().astype('float32')
            if idx_c < 4:
                pref[f'{name}__{c}__hist_min'] = v.groupby(b[best_key], sort=False).cummin().astype('float32')
                pref[f'{name}__{c}__hist_trend'] = (clast - cmean).astype('float32')
        pref = pref.drop_duplicates(subset=[best_key, '__t__'], keep='last').sort_values('__t__')
        a = pd.DataFrame({best_key: left_al[best_key], '__t__': lt, '__row__': np.arange(len(left))})
        valid = a[best_key].notna() & a['__t__'].notna()
        aok = a.loc[valid].sort_values('__t__')
        j = pd.merge_asof(aok, pref, on='__t__', by=best_key, direction='backward', allow_exact_matches=False)
        out = pd.concat([j, a.loc[~valid]], ignore_index=True).sort_values('__row__').reset_index(drop=True)
        # Causal RFM recency & activity rate (strictly past: __last_t__ < __t__)
        rec_days = (out['__t__'] - out['__last_t__']).dt.total_seconds() / 86400.0
        span_days = (out['__last_t__'] - out['__first_t__']).dt.total_seconds() / 86400.0
        out[f'{name}__hist_recency_days'] = rec_days.astype('float32')
        out[f'{name}__hist_span_days'] = span_days.astype('float32')
        out[f'{name}__hist_rate_per_day'] = (out[f'{name}__hist_count'] / np.maximum(span_days.fillna(0.0), 1.0)).astype('float32')
        win_cols = []
        # Fast O(N log N) strictly-causal sliding window counts/sums (1d, 7d, 30d, strictly < t) via searchsorted
        try:
            bk_str = b[best_key].astype('string')
            lk_str = left_al[best_key].astype('string')
            b_codes, uniques = pd.factorize(bk_str, sort=False)
            umap = {v: i for i, v in enumerate(uniques)}
            l_codes = lk_str.map(umap).fillna(-1).astype(np.int64).to_numpy()
            b_ns = b['__t__'].astype('int64').to_numpy()
            l_ns = lt.astype('int64').to_numpy()
            v_np = valid.to_numpy()
            t_min_ns = int(min(b_ns.min(), l_ns[v_np].min())) if v_np.any() else int(b_ns.min())
            b_sec = np.floor((b_ns - t_min_ns) / 1e9).astype(np.int64)
            l_sec = np.floor((l_ns - t_min_ns) / 1e9).astype(np.int64)
            max_sec = int(max(b_sec.max(), l_sec[v_np].max() if v_np.any() else 0))
            span_pad = max_sec + 2 * 30 * 86400 + 100_000
            b_key = (b_codes.astype(np.int64) + 1) * span_pad + b_sec
            b_ord = np.argsort(b_key, kind='mergesort')
            bk_s = b_key[b_ord]
            ok_q = v_np & (l_codes >= 0)
            q_idx = np.flatnonzero(ok_q)
            q_base = (l_codes[q_idx] + 1) * span_pad
            q_key = q_base + l_sec[q_idx]
            hi = np.searchsorted(bk_s, q_key, side='left')
            g_lo = np.searchsorted(bk_s, q_base, side='left')
            top_wcols = num_cols[:2]
            csums = {}
            for wc in top_wcols:
                vals_w = b[wc].to_numpy(dtype=np.float64)[b_ord]
                vals_w = np.where(np.isfinite(vals_w), vals_w, 0.0)
                csums[wc] = np.concatenate([[0.0], np.cumsum(vals_w)])
            w_cnts = {}
            for tag, w_sec in (('1d', 86400), ('7d', 7 * 86400), ('30d', 30 * 86400)):
                lo_w = np.maximum(np.searchsorted(bk_s, q_key - w_sec, side='left'), g_lo)
                cnt_w = (hi - lo_w).astype(np.float32)
                arr_c = np.zeros(len(left), dtype=np.float32)
                arr_c[q_idx] = cnt_w
                cname = f'{name}__hist_cnt_{tag}'
                out[cname] = arr_c
                win_cols.append(cname)
                w_cnts[tag] = arr_c
                if tag in ('7d', '30d'):
                    for wc in top_wcols:
                        sw = (csums[wc][hi] - csums[wc][lo_w]).astype(np.float32)
                        arr_s = np.zeros(len(left), dtype=np.float32)
                        arr_s[q_idx] = sw
                        sname = f'{name}__{wc}__hist_sum_{tag}'
                        out[sname] = arr_s
                        win_cols.append(sname)
                        if tag == '7d':
                            arr_m = np.full(len(left), np.nan, dtype=np.float32)
                            pos = cnt_w > 0
                            if pos.any():
                                arr_m[q_idx[pos]] = sw[pos] / cnt_w[pos]
                            mname = f'{name}__{wc}__hist_mean_7d'
                            out[mname] = arr_m
                            win_cols.append(mname)
            rname = f'{name}__hist_ratio_7d_30d'
            out[rname] = (w_cnts['7d'] / (w_cnts['30d'] + 1.0)).astype('float32')
            win_cols.append(rname)
        except Exception:
            log_exc(f'windowed causal RFM {name}')
        feat_cols = [c for c in pref.columns if c not in (best_key, '__t__', '__first_t__', '__last_t__')] + [
            f'{name}__hist_recency_days', f'{name}__hist_span_days', f'{name}__hist_rate_per_day'
        ] + win_cols
        out[f'{name}__hist_count'] = out[f'{name}__hist_count'].fillna(0).astype('float32')
        if (out[f'{name}__hist_count'] > 0).mean() == 0:
            return None
        return out[feat_cols], best_key, 'causal'
    # Static one-to-many table (no timestamps): safe groupby aggregation.
    g = dim_al.dropna(subset=[best_key]).groupby(best_key, sort=False)
    agg = pd.DataFrame(index=g.size().index)
    agg[f'{name}__count'] = g.size().astype('float32')
    for c in num_cols:
        agg[f'{name}__{c}__mean'] = g[c].mean().astype('float32')
        agg[f'{name}__{c}__std'] = g[c].std().astype('float32')
        agg[f'{name}__{c}__min'] = g[c].min().astype('float32')
        agg[f'{name}__{c}__max'] = g[c].max().astype('float32')
    for c in cat_cols:
        agg[f'{name}__{c}__nunique'] = g[c].nunique().astype('float32')
    merged = left_al[[best_key]].merge(agg, left_on=best_key, right_index=True, how='left', sort=False)
    feat_cols = list(agg.columns)
    if merged[f'{name}__count'].notna().mean() == 0:
        return None
    merged[f'{name}__count'] = merged[f'{name}__count'].fillna(0).astype('float32')
    return merged[feat_cols].reset_index(drop=True), best_key, 'static'


def _table_stems(name: str) -> list:
    """Generic plural-to-singular stem candidates for matching table PK `id` to `<stem>_id`."""
    low = name.lower().strip()
    stems = [low]
    if low.endswith('ies') and len(low) > 4:
        stems.append(low[:-3] + 'y')
    if low.endswith(('ses', 'xes', 'zes', 'ches', 'shes')) and len(low) > 4:
        stems.append(low[:-2])
    if low.endswith('s') and len(low) > 2:
        stems.append(low[:-1])
    return list(dict.fromkeys(stems))


def _norm_key_vals(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s):
        return s.astype('Int64').astype('string')
    if pd.api.types.is_float_dtype(s):
        ok = s.dropna()
        if len(ok) and np.all(np.isfinite(ok)) and np.all(ok == np.floor(ok)):
            return s.round().astype('Int64').astype('string')
    return s.astype('string').str.strip()


def _match_pk_to_fk(combined: pd.DataFrame, dim: pd.DataFrame, name: str, targets: set):
    """When dim has unique primary key `id` and combined has `<stem>_id`, match `dim.id` -> `combined.<stem>_id`."""
    id_col = next((c for c in dim.columns if c.lower() == 'id'), None)
    if id_col is None:
        return None, None
    stems = _table_stems(name)
    want = {f'{st}_id' for st in stems}
    fk_cands = [c for c in combined.columns if c not in targets and c not in dim.columns and c.lower() in want]
    if len(fk_cands) != 1:
        return None, None
    fk = fk_cands[0]
    pk_s = _norm_key_vals(dim[id_col])
    nonnull = pk_s.notna()
    valid = nonnull & ~pk_s.duplicated(keep=False)
    if valid.mean() < 0.95 or valid.sum() == 0:
        return None, None
    fk_s = _norm_key_vals(combined[fk]).dropna()
    if len(fk_s) == 0:
        return None, None
    pk_set = set(pk_s.loc[valid].tolist())
    overlap = float(fk_s.isin(pk_set).mean())
    if overlap < 0.2:
        return None, None
    return dim.rename(columns={id_col: fk}), fk


def enrich(train, test, tables, skip_names=(), report=None, meta=None):
    target_cols = [c for c in train.columns if c not in test.columns]
    targets = set(target_cols)
    n = len(train)
    # Keep targets completely outside combined concat so integer targets never upcast to float via test NaNs!
    train_targets = train[target_cols].reset_index(drop=True).copy()
    combined = pd.concat([train.drop(columns=target_cols), test], ignore_index=True)
    skip = {'train', 'test', 'sample_submission', 'data_dictionary'} | set(skip_names)
    remaining = {k: v for k, v in tables.items() if k not in skip}
    panels = []
    def _rec(msg):
        if isinstance(report, list):
            report.append(msg)
    # 0) Explicit PostgreSQL FOREIGN KEY joins (handles arbitrary column names and multi-role FKs like sender/receiver)
    fks = meta.get('foreign_keys', []) if isinstance(meta, dict) else []
    if fks:
        dst_counts = {}
        for fk in fks:
            dst_t = fk.get('dst_table')
            if dst_t in tables and dst_t not in skip:
                dst_counts[dst_t] = dst_counts.get(dst_t, 0) + 1
        applied_fks = set()
        fk_joined_dsts = set()
        for _ in range(3):
            fk_prog = False
            for fk in fks:
                src_c = fk.get('src_col')
                dst_t = fk.get('dst_table')
                dst_c = fk.get('dst_col')
                sig = (src_c, dst_t, dst_c)
                if sig in applied_fks or dst_t not in tables or dst_t in skip:
                    continue
                if src_c not in combined.columns or src_c in targets or dst_c not in tables[dst_t].columns:
                    continue
                try:
                    dim = tables[dst_t]
                    # Skip interval tables unless referencing their own PK (handled cleanly below)
                    _st_c = [c for c in dim.columns if c.lower().startswith('start_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
                    _en_c = [c for c in dim.columns if c.lower().startswith('end_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
                    if len(_st_c) == 1 and len(_en_c) == 1 and dim[dst_c].duplicated().any():
                        continue
                    dim_r = dim.rename(columns={dst_c: src_c}) if dst_c != src_c else dim
                    keys, dim_clean, excluded = _keys(dim_r, [src_c])
                    if not keys:
                        continue
                    multi_role = dst_counts.get(dst_t, 1) > 1
                    role_tag = re.sub(r'_id$', '', str(src_c), flags=re.I) if multi_role else ''
                    pfx = f'{role_tag}__{dst_t}' if (multi_role and role_tag and role_tag.lower() != dst_t.lower()) else dst_t
                    rename = {}
                    for c in dim_clean:
                        if c in keys or c in targets or POST_FACTUM_RE.search(str(c)):
                            continue
                        new_c = c if (not multi_role and c not in combined and c.endswith('_id')) else f'{pfx}__{c}'
                        if new_c not in combined.columns:
                            rename[c] = new_c
                    if not rename:
                        applied_fks.add(sig)
                        continue
                    right = dim_clean[keys + list(rename)].rename(columns=rename)
                    marker = f'{pfx}__matched'
                    if marker in combined.columns:
                        marker = f'{pfx}__{src_c}__matched'
                    right[marker] = 1.0
                    before = len(combined)
                    left_m, right_m = _align_key_dtypes(combined, right, keys)
                    candidate = left_m.merge(right_m, on=keys, how='left', sort=False, validate='many_to_one')
                    for k in keys:
                        candidate[k] = combined[k].values
                    cov = float(candidate[marker].notna().mean())
                    if cov > 0.0:
                        combined = candidate
                        assert len(combined) == before
                        combined[marker] = combined[marker].fillna(0)
                        log(f'join {dst_t} (SQL FK {src_c}->{dst_c}, pfx={pfx}): coverage={cov:.3f}, features={len(rename)}')
                        _rec(f'table {dst_t} ({len(dim)} rows): SQL FK {src_c}->{dst_c}, cov={cov:.3f}, +{len(rename)} feats')
                        applied_fks.add(sig)
                        fk_joined_dsts.add(dst_t)
                        fk_prog = True
                except Exception:
                    log_exc(f'SQL FK join {dst_t}')
                    applied_fks.add(sig)
            if not fk_prog:
                break
        for dst_t in fk_joined_dsts:
            remaining.pop(dst_t, None)
    for _ in range(4):
        changed = False
        for name, dim in list(remaining.items()):
            try:
                # Tables with (start_*, end_*) interval columns go through _interval_join unless they are an entity
                # dimension joined on its own primary key (e.g. campaigns on campaign_id).
                _st_c = [c for c in dim.columns if c.lower().startswith('start_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
                _en_c = [c for c in dim.columns if c.lower().startswith('end_') and any(t in c.lower() for t in ('time', 'date', 'ts', 'at'))]
                if len(_st_c) == 1 and len(_en_c) == 1:
                    _stems = _table_stems(name)
                    _own_pk = [c for c in dim.columns if c in combined and c not in targets
                               and c.lower() in {f'{st}_id' for st in _stems}]
                    if not (_own_pk and dim[_own_pk[0]].notna().all() and not dim[_own_pk[0]].duplicated().any()):
                        continue
                common = [c for c in dim if c in combined and c not in targets and
                          (c.lower().endswith('_id') or c.lower() == 'id' or c.lower() in TIME_NAMES or _is_time_col(c, dim[c]))]
                keys, dim_clean, excluded = None, None, 0
                if not common or [c.lower() for c in common] == ['id']:
                    dim_fk, fk_col = _match_pk_to_fk(combined, dim, name, targets)
                    if dim_fk is not None:
                        keys, dim_clean, excluded = _keys(dim_fk, [fk_col])
                if not keys:
                    keys, dim_clean, excluded = _keys(dim, common)
                if not keys:
                    continue
                if excluded:
                    log(f'join {name}: excluded {excluded} ambiguous/missing-key rows')
                rename = {c: (c if c not in combined and c.endswith('_id') else f'{name}__{c}')
                          for c in dim_clean if c not in keys and c not in targets}
                right = dim_clean[keys + list(rename)].rename(columns=rename)
                marker = f'{name}__matched'; right[marker] = 1.0
                before = len(combined)
                left_m, right_m = _align_key_dtypes(combined, right, keys)
                candidate = left_m.merge(right_m, on=keys, how='left', sort=False, validate='many_to_one')
                for k in keys:
                    candidate[k] = combined[k].values
                coverage = candidate[marker].notna().mean()
                mode_tag = 'many-to-one'
                if coverage < .95:
                    p_before = len(panels)
                    temporal = _temporal_join(combined, right, keys, marker,
                                              panels_out=panels if isinstance(meta, dict) else None,
                                              table_name=name)
                    if temporal is not None and temporal[0][marker].notna().mean() > coverage:
                        candidate, cadence = temporal
                        mode_tag = f'temporal-completed(shift={cadence})'
                        log(f'join {name}: completed-interval temporal lookup (strictly <= t - {cadence})')
                    elif len(panels) > p_before:
                        del panels[p_before:]
                has_time_key = any(_is_time_col(k, dim_clean[k]) for k in keys if k in dim_clean.columns)
                if candidate[marker].notna().mean() == 0 or (has_time_key and mode_tag == 'many-to-one' and coverage < 0.5):
                    iv = _interval_join(combined, dim_clean, name, targets)
                    if iv is not None:
                        combined = pd.concat([combined, iv], axis=1)
                        cov = float((iv[f'{name}__active_count'] > 0).mean())
                        log(f'join {name}: interval lookup, active_rate={cov:.4f}, features={iv.shape[1]}')
                        _rec(f'table {name} ({len(dim)} rows): interval join, active={cov:.3f}, +{iv.shape[1]} feats')
                        del remaining[name]; changed = True
                        continue
                    ag = _aggregate_child(combined, dim, name, targets)
                    if ag is not None:
                        feats_df, akey, amode = ag
                        combined = pd.concat([combined, feats_df], axis=1)
                        log(f'join {name}: one-to-many ({amode}) on {akey}, features={feats_df.shape[1]}')
                        _rec(f'table {name} ({len(dim)} rows): one-to-many {amode} on {akey}, +{feats_df.shape[1]} feats')
                        del remaining[name]; changed = True
                        continue
                    log(f'join skipped {name}: shared names but no matching key values')
                    del remaining[name]
                    continue
                combined = candidate
                assert len(combined) == before
                cov = float(combined[marker].notna().mean())
                log(f'join {name}: keys={keys}, coverage={cov:.3f}, features={len(rename)}')
                _rec(f'table {name} ({len(dim)} rows): {mode_tag} on {keys}, cov={cov:.3f}, +{len(rename)} feats')
                combined[marker] = combined[marker].fillna(0)
                del remaining[name]; changed = True
            except Exception:
                log_exc(f'join {name}')
                del remaining[name]
        if not changed: break
    for name in list(remaining):
        try:
            iv = _interval_join(combined, remaining[name], name, targets)
            if iv is not None:
                combined = pd.concat([combined, iv], axis=1)
                cov = float((iv[f'{name}__active_count'] > 0).mean())
                log(f'join {name}: interval lookup, active_rate={cov:.4f}, features={iv.shape[1]}')
                _rec(f'table {name} ({len(remaining[name])} rows): interval join, active={cov:.3f}, +{iv.shape[1]} feats')
                del remaining[name]
        except Exception:
            log_exc(f'interval join {name}')
    for name in list(remaining):
        try:
            ag = _aggregate_child(combined, remaining[name], name, targets)
            if ag is not None:
                feats_df, akey, amode = ag
                combined = pd.concat([combined, feats_df], axis=1)
                log(f'join {name}: one-to-many ({amode}) on {akey}, features={feats_df.shape[1]}')
                _rec(f'table {name} ({len(remaining[name])} rows): one-to-many {amode} on {akey}, +{feats_df.shape[1]} feats')
                del remaining[name]
        except Exception:
            log_exc(f'one-to-many {name}')
    for name in remaining:
        log(f'join skipped {name}: no unambiguous unique shared key')
    if isinstance(meta, dict):
        meta['strict_hours'] = True
        meta['panels'] = panels
        event_time = next((p.get('time_col') for p in panels if p.get('time_col') in combined.columns), None)
        meta['event_time'] = event_time
        if event_time is not None:
            meta['train_times'] = _parse_time_series(combined[event_time].iloc[:n]).reset_index(drop=True)
            meta['test_times'] = _parse_time_series(combined[event_time].iloc[n:]).reset_index(drop=True)
            gcols_all = sorted({g for p in panels for g in p['groups'] if g in combined.columns})
            if gcols_all:
                meta['train_groups'] = combined[gcols_all].iloc[:n].reset_index(drop=True)
                meta['test_groups'] = combined[gcols_all].iloc[n:].reset_index(drop=True)
    tr_out = pd.concat([combined.iloc[:n].reset_index(drop=True), train_targets], axis=1)
    te_out = combined.iloc[n:].reset_index(drop=True)
    return tr_out, te_out
