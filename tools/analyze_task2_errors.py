"""Pool 2.1: Residual error decomposition on hack_final_task2 (>5 min errors by storm phase)
and strictly causal A/B test of 07:00->08:00->09:00 precursors (d2 acceleration, 3h slope).
Run on server with PGDATABASE=hack_final_task2.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ag_database import extract
from ag_features import causal_stream_features, cross_impute_outages, detect_time_split, prepare
from ag_models import ModelConfig, fit_cat, fit_lgb, make_plan
from ag_nowcast import build_nowcast_features
from ag_relational import enrich
from ag_task import build_spec


def _build_precursors_07_08_09(tr_raw: pd.DataFrame, te_raw: pd.DataFrame, raw_tables: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly causal 2nd-order dynamics from completed hours (07:00 -> 08:00 -> 09:00 for 10:30 call)."""
    comb = pd.concat([tr_raw[['district_id', 'timestamp']], te_raw[['district_id', 'timestamp']]], ignore_index=True)
    comb['__t__'] = pd.to_datetime(comb['timestamp'], errors='coerce', utc=True)
    comb['__row__'] = np.arange(len(comb))
    n_tr = len(tr_raw)
    aok = comb.dropna(subset=['district_id', '__t__']).sort_values('__t__')
    out_parts = []
    for tname in ('weather', 'air_quality', 'power_grid', 'traffic', 'public_transport'):
        if tname not in raw_tables:
            continue
        df = raw_tables[tname].copy()
        df['__t__'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df = df.dropna(subset=['district_id', '__t__']).sort_values(['district_id', '__t__']).reset_index(drop=True)
        num_c = [c for c in df.columns if c not in ('district_id', 'timestamp', '__t__') and pd.api.types.is_numeric_dtype(df[c])]
        if not num_c:
            continue
        df[num_c] = df.groupby('district_id', sort=False)[num_c].ffill(limit=6)
        gb = df.groupby('district_id', sort=False)
        extra = {'district_id': df['district_id'], '__t__': df['__t__'] + pd.Timedelta(hours=1)}
        for c in num_c:
            s0 = df[c].astype('float32')
            s1 = gb[c].shift(1).astype('float32')
            s2 = gb[c].shift(2).astype('float32')
            extra[f'{tname}__{c}__d2'] = ((s0 - s1) - (s1 - s2)).astype('float32')
            extra[f'{tname}__{c}__slope3'] = ((s0 - s2) * 0.5).astype('float32')
        b = pd.DataFrame(extra).sort_values('__t__')
        j = pd.merge_asof(aok[['district_id', '__t__', '__row__']], b, on='__t__', by='district_id',
                          direction='backward', allow_exact_matches=True,
                          tolerance=pd.Timedelta(hours=1) - pd.Timedelta(nanoseconds=1))
        j = j.sort_values('__row__').reset_index(drop=True)
        fcols = [c for c in b.columns if c not in ('district_id', '__t__')]
        out_parts.append(j[fcols])
    all_p = pd.concat(out_parts, axis=1)
    return all_p.iloc[:n_tr].reset_index(drop=True), all_p.iloc[n_tr:].reset_index(drop=True)


def main():
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix='t2_err_'))
    raw = extract(work)
    tr_raw, te_raw = raw['train'].copy(), raw['test'].copy()
    dic = raw.get('data_dictionary')
    task_md = Path('/root/alfahack/dev/tasks/task2/task.md')
    meta = {}
    t_j, te_j = enrich(tr_raw, te_raw, raw, meta=meta)
    spec = build_spec(task_md, t_j, te_j, None, dic, {})
    st_tr, st_te = causal_stream_features(t_j, te_j, exclude={spec.target, spec.id_col})
    prep = prepare(pd.concat([t_j.reset_index(drop=True), st_tr], axis=1),
                   pd.concat([te_j.reset_index(drop=True), st_te], axis=1), spec)
    X, Xt = cross_impute_outages(prep.X, prep.Xt, prep.cat_cols)
    tcol = detect_time_split(X, Xt, prep.cat_cols)
    plan = make_plan(X, prep.y, 'regression', time_col=tcol)
    if meta.get('panels'):
        nc_tr, nc_te = build_nowcast_features(meta, prep.y, plan, 'regression', n_jobs=8, max_seconds=150.0)
        if nc_tr is not None:
            X = pd.concat([X.reset_index(drop=True), nc_tr.reset_index(drop=True)], axis=1)
            Xt = pd.concat([Xt.reset_index(drop=True), nc_te.reset_index(drop=True)], axis=1)

    print(f'Base features: X={X.shape}, prep time={time.time() - t0:.1f}s', flush=True)
    cfg_h = ModelConfig(objective_variant='huber')
    r_lgb0 = fit_lgb(X, prep.y, Xt, plan, spec, cfg_h, n_jobs=8, time_limit=180, lr=0.03, es=150, seed=0, name='lgb_huber_base')
    r_cat0 = fit_cat(X, prep.y, Xt, plan, spec, ModelConfig(), prep.cat_cols, n_jobs=8, time_limit=300, lr=0.06, depth=6, seed=0, name='cat_d6_base')
    print(f'[BASE] lgb_huber MAE={r_lgb0.score:.5f} folds={np.round(r_lgb0.fold_scores, 5).tolist()} ({r_lgb0.seconds:.1f}s)', flush=True)
    print(f'[BASE] cat_d6    MAE={r_cat0.score:.5f} folds={np.round(r_cat0.fold_scores, 5).tolist()} ({r_cat0.seconds:.1f}s)', flush=True)

    # Diagnostic-only storm phase decomposition on OOF rows (strictly offline diagnostic, NEVER a feature!)
    w_df = raw['weather'][['district_id', 'timestamp', 'storm_flag']].copy()
    w_df['hr'] = pd.to_datetime(w_df['timestamp'], utc=True)
    w_df = w_df.sort_values(['district_id', 'hr'])
    w_df['storm_prev'] = w_df.groupby('district_id')['storm_flag'].shift(1)
    tr_diag = tr_raw.loc[tr_raw[spec.target].notna()].reset_index(drop=True)[['district_id', 'timestamp']].copy()
    ts_utc = pd.to_datetime(tr_diag['timestamp'], utc=True)
    tr_diag['hr'] = ts_utc.dt.floor('h')
    tr_diag['minute'] = ts_utc.dt.minute
    tr_diag = tr_diag.merge(w_df[['district_id', 'hr', 'storm_flag', 'storm_prev']], on=['district_id', 'hr'], how='left')

    mask = plan.oof_mask & np.isfinite(r_cat0.oof)
    abs_err = np.abs(prep.y[mask] - r_cat0.oof[mask])
    sub = tr_diag.loc[mask].copy().reset_index(drop=True)
    sub['abs_err'] = abs_err
    sub['big'] = abs_err > 5.0

    def _bucket(r):
        cur, prev, mn = r.storm_flag, r.storm_prev, r.minute
        if cur == 1 and prev == 0:
            return 'storm_onset_0_14m' if mn < 15 else 'storm_onset_15_59m'
        if cur == 0 and prev == 1:
            return 'storm_offset'
        if cur == 1 and prev == 1:
            return 'storm_ongoing'
        return 'normal_hour'

    sub['phase'] = [ _bucket(r) for r in sub.itertuples(index=False) ]
    tot_err_sum = float(sub['abs_err'].sum())
    tot_big = int(sub['big'].sum())
    print(f'\n--- ERROR DECOMPOSITION (cat_d6 OOF n={len(sub)}, overall MAE={abs_err.mean():.5f}, |err|>5m count={tot_big} [{100*tot_big/len(sub):.2f}%]) ---')
    for phase, grp in sub.groupby('phase'):
        n_b = int(grp['big'].sum())
        print(f'  {phase:20s}: rows={len(grp):4d} ({100*len(grp)/len(sub):5.2f}%) | '
              f'MAE={grp["abs_err"].mean():6.3f} | share_total_MAE={100*grp["abs_err"].sum()/tot_err_sum:5.2f}% | '
              f'>5m={n_b:3d} (share_of_>5m={100*n_b/max(tot_big,1):5.2f}%)')

    # A/B test strictly causal 07:00->08:00->09:00 precursors (d2 + slope3)
    p_tr, p_te = _build_precursors_07_08_09(tr_raw.loc[tr_raw[spec.target].notna()].reset_index(drop=True), te_raw, raw)
    # Test subset (weather, air_quality, power_grid) and all 5 tables
    sub3_cols = [c for c in p_tr.columns if c.startswith(('weather__', 'air_quality__', 'power_grid__'))]
    for tag, cols_add in [('+d2_slope3_wap', sub3_cols), ('+d2_slope3_all5', list(p_tr.columns))]:
        X2 = pd.concat([X.reset_index(drop=True), p_tr[cols_add]], axis=1)
        Xt2 = pd.concat([Xt.reset_index(drop=True), p_te[cols_add]], axis=1)
        rl = fit_lgb(X2, prep.y, Xt2, plan, spec, cfg_h, n_jobs=8, time_limit=180, lr=0.03, es=150, seed=0, name=f'lgb_{tag}')
        rc = fit_cat(X2, prep.y, Xt2, plan, spec, ModelConfig(), prep.cat_cols, n_jobs=8, time_limit=300, lr=0.06, depth=6, seed=0, name=f'cat_{tag}')
        print(f'[{tag}] (+{len(cols_add)} cols) lgb_huber={rl.score:.5f} (d={rl.score-r_lgb0.score:+.5f}) | '
              f'cat_d6={rc.score:.5f} (d={rc.score-r_cat0.score:+.5f})', flush=True)


if __name__ == '__main__':
    main()
