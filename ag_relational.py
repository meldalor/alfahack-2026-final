"""Schema-driven joins: unique dimensions, bounded temporal lookup, intervals, and causal one-to-many."""
import itertools
import numpy as np
import pandas as pd
from ag_utils import log, log_exc

TIME_NAMES = {'timestamp', 'date', 'datetime', 'time', 'event_time', 'created_at', 'updated_at', 'recorded_at', 'ts', 'dt'}


def _is_time_col(name: str, series: pd.Series | None = None) -> bool:
    low = str(name).lower()
    if low.startswith(('start_', 'end_')):
        return False
    name_ok = (low in TIME_NAMES or low.endswith(('_time', '_timestamp', '_date', '_datetime', '_at', '_ts', '_dt'))
               or low.startswith(('timestamp_', 'datetime_', 'event_time')))
    if not name_ok:
        return False
    if series is None:
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().head(200)
    if len(sample) < 3:
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parsed = pd.to_datetime(sample, errors='coerce', utc=True)
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


def _temporal_join(left, right, keys, marker):
    times = [c for c in keys if _is_time_col(c, right[c])]
    if len(times) != 1:
        return None
    tc = times[0]
    groups = [c for c in keys if c != tc]
    left, right = _align_key_dtypes(left, right, groups)
    rt = pd.to_datetime(right[tc], errors='coerce', utc=True)
    lt = pd.to_datetime(left[tc], errors='coerce', utc=True)
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
    # Causal temporal dynamics from strictly completed past observations (all shifts >= 1, strictly <= rt <= ev - cadence).
    n_groups = int(b.groupby(groups, dropna=False).ngroups) if groups else 1
    per_grp = len(b) / max(n_groups, 1)
    if base_num and per_grp >= 12:
        b = b.sort_values((groups + [key]) if groups else [key]).reset_index(drop=True)
        gb = b.groupby(groups, sort=False) if groups else None
        extra = {}
        is_hourly = (cadence == pd.Timedelta(hours=1))
        is_daily = (cadence == pd.Timedelta(days=1))
        for c in base_num[:16]:
            s = b[c].astype('float32')
            s1 = (gb[c].shift(1) if gb is not None else s.shift(1)).astype('float32')
            extra[f'{c}__d1'] = (s - s1).astype('float32')
            rm3 = (gb[c].transform(lambda x: x.rolling(3, min_periods=1).mean())
                   if gb is not None else s.rolling(3, min_periods=1).mean())
            extra[f'{c}__rm3'] = rm3.astype('float32')
            # Strictly causal run-length of active binary state (consecutive 1s ending at completed interval rt).
            non_na = b[c].dropna()
            if len(non_na) >= 10 and non_na.iloc[:max(10, len(non_na) // 2)].nunique() == 2 and non_na.isin([0, 1, 0.0, 1.0]).mean() >= 0.95:
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
    groups = [c for c in dim.columns if c in left.columns and c not in targets
              and (c.lower().endswith('_id') or c.lower() == 'id')
              and not (c.lower() == 'event_id' and left[c].nunique() > 0.9 * len(left))]
    if not groups or len(dim) > 50_000:
        return None
    lt = pd.to_datetime(left[ltc], errors='coerce', utc=True)
    st = pd.to_datetime(dim[sc], errors='coerce', utc=True)
    et = pd.to_datetime(dim[ec], errors='coerce', utc=True)
    if lt.notna().mean() < 0.9 or st.notna().mean() < 0.9 or et.notna().mean() < 0.9:
        return None
    a = left[groups].copy(); a['__t__'] = lt; a['__row__'] = np.arange(len(left))
    dim_lows = {c.lower() for c in dim.columns}
    keep_cols = [c for c in dim.columns if c not in groups and c not in (sc, ec) and c not in targets
                 and not c.lower().endswith('_id') and c.lower() != 'id'
                 and not (c.lower().startswith('actual_') and ('expected_' + c.lower()[7:]) in dim_lows)]
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
                and pd.api.types.is_numeric_dtype(dim_al[c])][:8]
    cat_cols = [c for c in dim_al.columns if c != best_key and c not in targets and c not in d_times
                and c not in num_cols and not c.lower().endswith('_id') and c.lower() != 'id'
                and 2 <= dim_al[c].nunique(dropna=True) <= 200][:4]
    if d_times:
        if len(d_times) != 1 or len(l_times) != 1:
            return None  # Timed child table without matching event clock: skip to prevent future leakage.
        dtc, ltc = d_times[0], l_times[0]
        dt = pd.to_datetime(dim_al[dtc], errors='coerce', utc=True)
        lt = pd.to_datetime(left[ltc], errors='coerce', utc=True)
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
        feat_cols = [c for c in pref.columns if c not in (best_key, '__t__', '__first_t__', '__last_t__')] + [
            f'{name}__hist_recency_days', f'{name}__hist_span_days', f'{name}__hist_rate_per_day'
        ]
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


def enrich(train, test, tables, skip_names=(), report=None):
    targets = set(train) - set(test)
    n = len(train)
    combined = pd.concat([train, test], ignore_index=True)
    skip = {'train', 'test', 'sample_submission', 'data_dictionary'} | set(skip_names)
    remaining = {k: v for k, v in tables.items() if k not in skip}
    def _rec(msg):
        if isinstance(report, list):
            report.append(msg)
    for _ in range(4):
        changed = False
        for name, dim in list(remaining.items()):
            try:
                common = [c for c in dim if c in combined and c not in targets and
                          (c.lower().endswith('_id') or c.lower() == 'id' or c.lower() in TIME_NAMES or _is_time_col(c, dim[c]))]
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
                    temporal = _temporal_join(combined, right, keys, marker)
                    if temporal is not None and temporal[0][marker].notna().mean() > coverage:
                        candidate, cadence = temporal
                        mode_tag = f'temporal-completed(shift={cadence})'
                        log(f'join {name}: completed-interval temporal lookup (strictly <= t - {cadence})')
                if candidate[marker].notna().mean() == 0:
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
    return combined.iloc[:n].reset_index(drop=True), combined.iloc[n:].drop(columns=list(targets)).reset_index(drop=True)
