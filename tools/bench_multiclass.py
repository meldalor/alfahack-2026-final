"""Pool 3.4 benchmark: synthetic 15-class multiclass (50k rows, 50 num + 10 cat features).
Measures runtime of all model families + simplex stacking + strict time_limit abort behavior.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ag_ensemble import blend, stack_blend
from ag_models import (ModelConfig, fit_cat, fit_et, fit_hgb, fit_lgb, fit_linear, fit_mlp, fit_xgb, make_plan)
from ag_task import TaskSpec


def main():
    rng = np.random.default_rng(7)
    n_tr, n_te, n_classes = 50_000, 10_000, 15
    Z = rng.normal(size=(n_tr + n_te, 50)).astype(np.float32)
    W = rng.normal(0, 0.4, size=(50, n_classes)).astype(np.float32)
    logits = Z[:n_tr] @ W + rng.normal(0, 0.8, size=(n_tr, n_classes))
    y = logits.argmax(axis=1).astype(int)

    df = pd.DataFrame(Z, columns=[f'f{i}' for i in range(50)])
    for j in range(10):
        card = 5 + 4 * j
        vals = [f'c{j}_{k}' for k in range(card)]
        df[f'cat{j}'] = pd.Categorical(rng.choice(vals, size=n_tr + n_te))
    X, Xt = df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True)
    cat_cols = [f'cat{j}' for j in range(10)]
    spec = TaskSpec('', '', id_col='id', target='y', task_type='multiclass', metric='logloss', classes=list(range(n_classes)))
    plan = make_plan(X, y, 'multiclass', n_folds=3, seed=42)
    cfg = ModelConfig()
    print(f'Multiclass stress: X={X.shape}, classes={n_classes}, folds={len(plan.folds)}', flush=True)

    results = []
    runners = [
        ('lgb', lambda lim: fit_lgb(X, y, Xt, plan, spec, cfg, 8, lim, lr=0.08, es=50, seed=0, name='lgb')),
        ('logreg', lambda lim: fit_linear(X, y, Xt, plan, spec, cfg, 8, lim, kind='logreg', name='logreg')),
        ('xgb', lambda lim: fit_xgb(X, y, Xt, plan, spec, cfg, 8, lim, lr=0.08, depth=6, es=50, name='xgb')),
        ('hgb', lambda lim: fit_hgb(X, y, Xt, plan, spec, cfg, 8, lim, lr=0.08, max_iter=150, name='hgb')),
        ('cat', lambda lim: fit_cat(X, y, Xt, plan, spec, cfg, cat_cols, 8, lim, lr=0.12, depth=6, es=50, name='cat')),
        ('et', lambda lim: fit_et(X, y, Xt, plan, spec, cfg, 8, lim, n_estimators=150, name='et')),
        ('mlp', lambda lim: fit_mlp(X, y, Xt, plan, spec, cfg, 8, lim, name='mlp')),
    ]
    print('\n--- FULL SPEED BENCHMARK (time_limit=180s per family) ---')
    for nm, fn in runners:
        t0 = time.time()
        r = fn(180.0)
        dt = time.time() - t0
        if r is None:
            print(f'  {nm:8s}: SKIPPED/NONE in {dt:5.1f}s', flush=True)
        else:
            print(f'  {nm:8s}: logloss={r.score:.5f}, iters={r.iters}, time={dt:5.1f}s, cut={r.config.get("time_cut", False)}', flush=True)
            results.append(r)

    if len(results) >= 3:
        t_s = time.time()
        b = blend(results, y, plan.oof_mask, spec, plan=plan)
        sb = stack_blend(results, y, plan.oof_mask, spec, plan=plan)
        from ag_metrics import proba_score
        print(f'\nSimplex blend ({time.time()-t_s:.2f}s): logloss={proba_score("logloss", y, b["oof"]):.5f} weights={b["weights"]}')
        print(f'Simplex stack: logloss={proba_score("logloss", y, sb["oof"]):.5f} weights={sb["weights"]}', flush=True)

    print('\n--- TIGHT TIME-LIMIT CHECK (time_limit=10s) ---')
    for nm, fn in runners:
        t0 = time.time()
        r = fn(10.0)
        dt = time.time() - t0
        status = 'SKIPPED' if r is None else f'DONE/CUT(cut={r.config.get("time_cut", False)}, score={r.score:.4f})'
        print(f'  {nm:8s} (limit=10s): {status} in {dt:5.1f}s', flush=True)


if __name__ == '__main__':
    main()
