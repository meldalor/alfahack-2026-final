"""Pool 3.2 & 3.3 benchmark on hack_final_task2 (3-fold chronological CV):
1) Multi-loss GBDT comparison (LGB L1/Huber/L2, XGB L1/PseudoHuber, CatBoost MAE/RMSE/Huber, HGB L1)
2) Tabular Neural Net comparison: sklearn MLP (L2) vs NumPy ResNet (SmoothL1/L1) vs PyTorch ResNet (if torch available)
3) Full simplex ensemble & nested OOF stack_blend impact.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ag_database import extract
from ag_ensemble import blend, stack_blend
from ag_features import causal_stream_features, cross_impute_outages, detect_time_split, prepare
from ag_metrics import score
from ag_models import (FitResult, ModelConfig, fit_cat, fit_hgb, fit_lgb, fit_mlp, fit_xgb, make_plan)
from ag_nowcast import build_nowcast_features
from ag_relational import enrich
from ag_task import build_spec


def _fit_sklearn_mlp_baseline(X, y, Xt, plan, spec, top_cols, max_cols=80):
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

    t0 = time.time()
    cols = [c for c in X.columns if c != plan.time_col and not str(c).endswith(("__ord", "__year"))]
    if top_cols:
        cols = [c for c in top_cols if c in cols][:max_cols]
    cat_cols = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object or str(X[c].dtype) == "string"]
    num_cols = [c for c in cols if c not in cat_cols]
    X_in, Xt_in = X[cols].copy(), Xt[cols].copy()
    for c in cat_cols:
        X_in[c] = X_in[c].astype("string").fillna("__missing__").astype(str)
        Xt_in[c] = Xt_in[c].astype("string").fillna("__missing__").astype(str)
    for c in num_cols:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        Xt_in[c] = pd.to_numeric(Xt_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_lo, y_hi = float(np.nanmin(y)), float(np.nanmax(y))
    oof = np.full(len(X), np.nan)
    for tr, va in plan.folds:
        tf = []
        if num_cols:
            tf.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                            QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0)), num_cols))
        if cat_cols:
            tf.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=40, sparse_output=False), cat_cols))
        m = make_pipeline(ColumnTransformer(tf, sparse_threshold=0.0),
                          TransformedTargetRegressor(
                              regressor=MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-2,
                                                     batch_size=256, learning_rate_init=1e-3, max_iter=80,
                                                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=0),
                              transformer=StandardScaler()))
        m.fit(X_in.iloc[tr], y[tr])
        oof[va] = np.clip(m.predict(X_in.iloc[va]), y_lo, y_hi)
    s = score(spec.metric, "regression", y[plan.oof_mask], oof[plan.oof_mask])
    return FitResult("mlp_sklearn_l2", oof, np.zeros(len(Xt)), s, [], [], None, time.time() - t0)


def _fit_torch_resnet(X, y, Xt, plan, spec, top_cols, loss_name="smooth_l1", max_cols=80):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    t0 = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cols = [c for c in X.columns if c != plan.time_col and not str(c).endswith(("__ord", "__year"))]
    if top_cols:
        cols = [c for c in top_cols if c in cols][:max_cols]
    cat_cols = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object or str(X[c].dtype) == "string"]
    num_cols = [c for c in cols if c not in cat_cols]
    X_in, Xt_in = X[cols].copy(), Xt[cols].copy()
    for c in cat_cols:
        X_in[c] = X_in[c].astype("string").fillna("__missing__").astype(str)
        Xt_in[c] = Xt_in[c].astype("string").fillna("__missing__").astype(str)
    for c in num_cols:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        Xt_in[c] = pd.to_numeric(Xt_in[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_lo, y_hi = float(np.nanmin(y)), float(np.nanmax(y))
    oof = np.full(len(X), np.nan)

    class TabResNet(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.in_proj = nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.12))
            self.res = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.12))
            self.head = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.08), nn.Linear(64, 1))

        def forward(self, x):
            h = self.in_proj(x)
            h = h + self.res(h)
            return self.head(h).squeeze(-1)

    for k, (tr, va) in enumerate(plan.folds):
        tf = []
        if num_cols:
            tf.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                            QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=k)), num_cols))
        if cat_cols:
            tf.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=40, sparse_output=False), cat_cols))
        pre = ColumnTransformer(tf, sparse_threshold=0.0)
        Ztr = np.nan_to_num(np.asarray(pre.fit_transform(X_in.iloc[tr]), dtype=np.float32), 0.0)
        Zva = np.nan_to_num(np.asarray(pre.transform(X_in.iloc[va]), dtype=np.float32), 0.0)
        ym, ys = float(np.median(y[tr])), float(np.std(y[tr])) or 1.0
        yn = ((y[tr] - ym) / ys).astype(np.float32)

        rng = np.random.default_rng(k)
        perm = rng.permutation(len(Ztr))
        nv = max(64, int(0.1 * len(Ztr)))
        iv, it = perm[:nv], perm[nv:]
        net = TabResNet(Ztr.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=65)
        crit = nn.L1Loss() if loss_name == "l1" else nn.SmoothL1Loss(beta=0.35)
        xt = torch.from_numpy(Ztr[it]); yt_t = torch.from_numpy(yn[it])
        xv = torch.from_numpy(Ztr[iv]); yv_t = torch.from_numpy(yn[iv])
        best_l, best_st, no_imp = float("inf"), None, 0
        for ep in range(65):
            net.train()
            p_ep = torch.randperm(len(xt))
            for i in range(0, len(xt), 256):
                bi = p_ep[i:i + 256]
                opt.zero_grad()
                loss = crit(net(xt[bi]), yt_t[bi])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vl = float(crit(net(xv), yv_t).item())
            if vl < best_l - 1e-5:
                best_l, best_st, no_imp = vl, {kk: vv.clone() for kk, vv in net.state_dict().items()}, 0
            else:
                no_imp += 1
                if no_imp >= 10:
                    break
        if best_st is not None:
            net.load_state_dict(best_st)
        net.eval()
        with torch.no_grad():
            pred_va = net(torch.from_numpy(Zva)).numpy() * ys + ym
        oof[va] = np.clip(pred_va, y_lo, y_hi)
    s = score(spec.metric, "regression", y[plan.oof_mask], oof[plan.oof_mask])
    return FitResult(f"mlp_torch_{loss_name}", oof, np.zeros(len(Xt)), s, [], [], None, time.time() - t0)


def main():
    work = Path(tempfile.mkdtemp(prefix='p3_bench_'))
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

    print(f'Prepared X={X.shape}, folds={len(plan.folds)}', flush=True)
    # Multi-loss GBDT suite
    r_lgb_hub = fit_lgb(X, prep.y, Xt, plan, spec, ModelConfig(objective_variant='huber'), 8, 180, lr=0.03, es=150, seed=0, name='lgb_huber')
    r_lgb_l1 = fit_lgb(X, prep.y, Xt, plan, spec, ModelConfig(), 8, 180, lr=0.03, es=150, seed=0, name='lgb_l1')
    r_lgb_l2 = fit_lgb(X, prep.y, Xt, plan, spec, ModelConfig(objective_variant='l2'), 8, 180, lr=0.03, es=150, seed=0, name='lgb_l2')
    top_cols = list(r_lgb_hub.importance.sort_values(ascending=False).index)

    r_xgb_ph = fit_xgb(X, prep.y, Xt, plan, spec, ModelConfig(), 8, 240, lr=0.04, depth=6, objective_override='pseudohuber', name='xgb_ph')
    r_hgb = fit_hgb(X, prep.y, Xt, plan, spec, ModelConfig(), 8, 180, top_cols=top_cols, name='hgb')
    r_cat_mae = fit_cat(X, prep.y, Xt, plan, spec, ModelConfig(), prep.cat_cols, 8, 300, lr=0.06, depth=6, seed=0, name='cat_d6_mae')
    r_cat_rmse = fit_cat(X, prep.y, Xt, plan, spec, ModelConfig(), prep.cat_cols, 8, 300, lr=0.06, depth=6, seed=2, loss_override='RMSE', name='cat_d6_rmse')
    r_cat_hub = fit_cat(X, prep.y, Xt, plan, spec, ModelConfig(), prep.cat_cols, 8, 300, lr=0.06, depth=6, seed=3, loss_override='Huber', name='cat_d6_huber')
    r_cat_d7 = fit_cat(X, prep.y, Xt, plan, spec, ModelConfig(), prep.cat_cols, 8, 360, lr=0.06, depth=7, seed=1, name='cat_d7_mae')

    # Neural net comparison
    r_mlp_sk = _fit_sklearn_mlp_baseline(X, prep.y, Xt, plan, spec, top_cols)
    r_mlp_np = fit_mlp(X, prep.y, Xt, plan, spec, ModelConfig(), 8, 180, seed=0, top_cols=top_cols, name='mlp_numpy_huber')
    r_mlp_pt_s = _fit_torch_resnet(X, prep.y, Xt, plan, spec, top_cols, loss_name='smooth_l1')
    r_mlp_pt_l1 = _fit_torch_resnet(X, prep.y, Xt, plan, spec, top_cols, loss_name='l1')

    all_models = [r_lgb_hub, r_lgb_l1, r_lgb_l2, r_xgb_ph, r_hgb, r_cat_mae, r_cat_rmse, r_cat_hub, r_cat_d7, r_mlp_sk, r_mlp_np]
    if r_mlp_pt_s is not None:
        all_models.extend([r_mlp_pt_s, r_mlp_pt_l1])

    print('\n--- SOLO 3-FOLD CV MAE ---')
    for r in all_models:
        if r is not None:
            print(f'  {r.name:20s}: MAE={r.score:.5f} ({r.seconds:5.1f}s)')

    gbdt_core = [r_cat_d7, r_cat_rmse, r_cat_mae, r_lgb_hub, r_xgb_ph, r_hgb]
    print('\n--- ENSEMBLE COMPARISON (SIMPLEX BLEND & NESTED STACK) ---')
    variants = [
        ('GBDT_only', gbdt_core),
        ('GBDT+cat_hub+lgb_l2', gbdt_core + [r_cat_hub, r_lgb_l2]),
        ('GBDT+mlp_sklearn_l2', gbdt_core + [r_mlp_sk]),
        ('GBDT+mlp_numpy_huber', gbdt_core + [r_mlp_np]),
    ]
    if r_mlp_pt_s is not None:
        variants.append(('GBDT+mlp_torch_smoothl1', gbdt_core + [r_mlp_pt_s]))
        variants.append(('GBDT+mlp_torch_l1', gbdt_core + [r_mlp_pt_l1]))
    for tag, grp in variants:
        b = blend(grp, prep.y, plan.oof_mask, spec, plan=plan)
        sb = stack_blend(grp, prep.y, plan.oof_mask, spec, plan=plan)
        print(f'  {tag:24s}: blend={b["score"]:.5f} {b["weights"]} | nested_stack={sb["score"]:.5f} {sb["weights"]}', flush=True)


if __name__ == '__main__':
    main()
