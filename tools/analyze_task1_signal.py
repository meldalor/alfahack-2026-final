"""Pool 2.2 & 2.3: Independent signal audit on hack_final_task1 (PR-AUC across chronological folds)
and speed benchmark of cross-table tag + categorical matching on 400k rows.
Run on server with PGDATABASE=hack_final_task1.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ag_database import extract
from ag_features import (causal_stream_features, detect_time_split, entity_periods,
                         op_entity_history, prepare)
from ag_models import ModelConfig, fit_hgb, fit_lgb, fit_xgb, make_plan
from ag_relational import enrich
from ag_task import build_spec


def main():
    work = Path(tempfile.mkdtemp(prefix='t1_audit_'))
    t0 = time.time()
    raw = extract(work)
    print(f'extract: {time.time() - t0:.2f}s, train={raw["train"].shape}, test={raw["test"].shape}', flush=True)

    t1 = time.time()
    t_j, te_j = enrich(raw['train'], raw['test'], raw)
    print(f'enrich: {time.time() - t1:.2f}s -> train={t_j.shape}, test={te_j.shape}', flush=True)

    task_md = Path('/root/alfahack/dev/tasks/task1/task.md')
    spec = build_spec(task_md, t_j, te_j, None, None, {})

    t2 = time.time()
    st_tr, st_te = causal_stream_features(t_j, te_j, exclude={spec.target, spec.id_col})
    print(f'causal_stream: {time.time() - t2:.2f}s -> +{st_tr.shape[1]} feats', flush=True)

    t3 = time.time()
    prep = prepare(pd.concat([t_j.reset_index(drop=True), st_tr], axis=1),
                   pd.concat([te_j.reset_index(drop=True), st_te], axis=1), spec)
    dt_prep = time.time() - t3
    match_cols = [c for c in prep.X.columns if any(k in c for k in ('__ov__', '__jac__', '__eq__', '__n_tags'))]
    print(f'prepare (incl. cross-table match on {len(prep.X)+len(prep.Xt)} rows): {dt_prep:.2f}s, match_cols={match_cols}', flush=True)

    # Pure timing of matching only (isolated)
    tcol = detect_time_split(prep.X, prep.Xt, prep.cat_cols)
    plan = make_plan(prep.X, prep.y, 'binary', time_col=tcol)
    print(f'time_col={tcol}, plan={plan.desc}', flush=True)

    # Check univariate correlation / positive rates of match features on train
    print('\n--- UNIVARIATE SIGNAL CHECK (TRAIN) ---')
    base_rate = float(prep.y.mean())
    print(f'Base positive rate engaged=1: {base_rate:.5f}')
    for c in match_cols:
        s = prep.X[c]
        g = pd.DataFrame({'v': s, 'y': prep.y}).groupby('v', dropna=False)['y'].agg(['count', 'mean'])
        print(f'  {c}:')
        for val, row in g.head(8).iterrows():
            print(f'    val={val}: n={int(row["count"])}, mean_y={row["mean"]:.5f} (diff={row["mean"]-base_rate:+.5f})')

    # Ablation on chronological 2-fold CV:
    # A) Base enriched without stream and without match
    no_stream_no_match = [c for c in prep.X.columns if not c.startswith(('stream__', 'stream_grp__')) and c not in match_cols]
    # B) Base + cross-table match (interests ov/jac + gender/age __eq__)
    base_plus_match = [c for c in prep.X.columns if not c.startswith(('stream__', 'stream_grp__'))]
    # C) Full (base + match + stream + stream_grp user_id/campaign_id)
    full_cols = list(prep.X.columns)

    # D) Full + causal entity history on user_id and campaign_id
    t_tr, t_te = entity_periods(prep.X, prep.Xt, tcol)
    eh_u_tr, eh_u_te = op_entity_history(prep.X, prep.Xt, prep.y, ['user_id'], tcol, t_tr, t_te)
    eh_c_tr, eh_c_te = op_entity_history(prep.X, prep.Xt, prep.y, ['campaign_id'], tcol, t_tr, t_te)
    X_eh = pd.concat([prep.X, eh_u_tr.add_prefix('u_'), eh_c_tr.add_prefix('c_')], axis=1)
    Xt_eh = pd.concat([prep.Xt, eh_u_te.add_prefix('u_'), eh_c_te.add_prefix('c_')], axis=1)

    print('\n--- CHRONOLOGICAL CV ABLATION (PR-AUC) ---')
    for label, Xf, Xtf in [
        ('A_base_only', prep.X[no_stream_no_match], prep.Xt[no_stream_no_match]),
        ('B_base+match', prep.X[base_plus_match], prep.Xt[base_plus_match]),
        ('C_full+stream', prep.X[full_cols], prep.Xt[full_cols]),
        ('D_full+entity_hist', X_eh, Xt_eh),
    ]:
        rl = fit_lgb(Xf, prep.y, Xtf, plan, spec, ModelConfig(), n_jobs=8, time_limit=120, lr=0.05, es=50, seed=0, name=label)
        print(f'  LGB {label:20s} ({Xf.shape[1]:3d} feats): PR-AUC={rl.score:.5f} folds={np.round(rl.fold_scores, 5).tolist()} iters={rl.iters} ({rl.seconds:.1f}s)', flush=True)

    rx = fit_xgb(prep.X, prep.y, prep.Xt, plan, spec, ModelConfig(), n_jobs=8, time_limit=180, lr=0.08, depth=6, es=50, seed=0, name='xgb_full')
    rh = fit_hgb(prep.X, prep.y, prep.Xt, plan, spec, ModelConfig(), n_jobs=8, time_limit=180, max_iter=200, seed=0, name='hgb_full')
    print(f'  XGB full: PR-AUC={rx.score:.5f} folds={np.round(rx.fold_scores, 5).tolist()} ({rx.seconds:.1f}s)', flush=True)
    print(f'  HGB full: PR-AUC={rh.score:.5f} folds={np.round(rh.fold_scores, 5).tolist()} ({rh.seconds:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
