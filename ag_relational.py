"""Schema-driven joins: unique dimensions and backward, bounded temporal lookup."""
import itertools
import numpy as np
import pandas as pd
from ag_utils import log

TIME_NAMES = {'timestamp', 'date', 'datetime'}


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
    times = [c for c in keys if c.lower() in TIME_NAMES]
    if len(times) != 1:
        return None
    tc = times[0]
    groups = [c for c in keys if c != tc]
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
    fill_cols = [c for c in b.columns if c not in groups and c not in (key, marker)
                 and pd.api.types.is_numeric_dtype(b[c]) and b[c].isna().any()]
    if fill_cols:
        null_col = f'{marker[:-len("__matched")]}__raw_nulls' if marker.endswith('__matched') else f'{marker}__raw_nulls'
        b[null_col] = b[fill_cols].isna().sum(axis=1).astype('float32')
        if groups:
            b[fill_cols] = b.groupby(groups, sort=False)[fill_cols].ffill(limit=6)
        else:
            b[fill_cols] = b[fill_cols].ffill(limit=6)
    # Exclude the boundary of the following interval: an hour-old reading is stale.
    joined = pd.merge_asof(aok, b, on=key, by=groups or None, direction='backward',
                           tolerance=cadence - pd.Timedelta(nanoseconds=1))
    missing = a.loc[~valid]
    joined = pd.concat([joined, missing], ignore_index=True).sort_values(row)
    joined = joined.drop(columns=[key, row]).reset_index(drop=True)
    return joined, cadence


def _interval_join(left, dim, name, targets):
    """Generic interval join for tables with (start_*, end_*) timestamps and shared group keys."""
    l_times = [c for c in left.columns if c.lower() in TIME_NAMES]
    if len(l_times) != 1:
        return None
    ltc = l_times[0]
    starts = [c for c in dim.columns if c.lower().startswith('start_') and any(t in c.lower() for t in ('time', 'date'))]
    ends = [c for c in dim.columns if c.lower().startswith('end_') and any(t in c.lower() for t in ('time', 'date'))]
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
    keep_cols = [c for c in dim.columns if c not in groups and c not in (sc, ec) and c not in targets
                 and not c.lower().endswith('_id') and c.lower() != 'id']
    b = dim[groups + keep_cols].copy(); b['__s__'] = st; b['__e__'] = et
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


def enrich(train, test, tables):
    targets = set(train) - set(test)
    n = len(train)
    combined = pd.concat([train, test], ignore_index=True)
    remaining = {k: v for k, v in tables.items()
                 if k not in ('train', 'test', 'sample_submission', 'data_dictionary')}
    for _ in range(4):
        changed = False
        for name, dim in list(remaining.items()):
            common = [c for c in dim if c in combined and c not in targets and
                      (c.lower().endswith('_id') or c.lower() == 'id' or c.lower() in TIME_NAMES)]
            keys, dim, excluded = _keys(dim, common)
            if not keys:
                continue
            if excluded:
                log(f'join {name}: excluded {excluded} ambiguous/missing-key rows')
            rename = {c: (c if c not in combined and c.endswith('_id') else f'{name}__{c}')
                      for c in dim if c not in keys and c not in targets}
            right = dim[keys + list(rename)].rename(columns=rename)
            marker = f'{name}__matched'; right[marker] = 1.0
            before = len(combined)
            candidate = combined.merge(right, on=keys, how='left', sort=False, validate='many_to_one')
            coverage = candidate[marker].notna().mean()
            if coverage < .95:
                temporal = _temporal_join(combined, right, keys, marker)
                if temporal is not None and temporal[0][marker].notna().mean() > coverage:
                    candidate, cadence = temporal
                    log(f'join {name}: backward temporal lookup, maximum age < {cadence}')
            if candidate[marker].notna().mean() == 0:
                iv = _interval_join(combined, dim, name, targets)
                if iv is not None:
                    combined = pd.concat([combined, iv], axis=1)
                    cov = float((iv[f'{name}__active_count'] > 0).mean())
                    log(f'join {name}: interval lookup, active_rate={cov:.4f}, features={iv.shape[1]}')
                    del remaining[name]; changed = True
                    continue
                log(f'join skipped {name}: shared names but no matching key values')
                del remaining[name]
                continue
            combined = candidate
            assert len(combined) == before
            log(f'join {name}: keys={keys}, coverage={combined[marker].notna().mean():.3f}, features={len(rename)}')
            combined[marker] = combined[marker].fillna(0)
            del remaining[name]; changed = True
        if not changed: break
    for name in list(remaining):
        iv = _interval_join(combined, remaining[name], name, targets)
        if iv is not None:
            combined = pd.concat([combined, iv], axis=1)
            cov = float((iv[f'{name}__active_count'] > 0).mean())
            log(f'join {name}: interval lookup, active_rate={cov:.4f}, features={iv.shape[1]}')
            del remaining[name]
    for name in remaining:
        log(f'join skipped {name}: no unambiguous unique shared key')
    return combined.iloc[:n].reset_index(drop=True), combined.iloc[n:].drop(columns=list(targets)).reset_index(drop=True)
