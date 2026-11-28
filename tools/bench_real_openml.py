"""Real-world 60-dataset OpenML benchmark + Smart Type-Aware Chaos Corruptor + Release Scorecard.

Offline validation harness ONLY (excluded from production ZIP).
Covers 12 narrow supervised tabular/DB ML task types x 5 verified real-world OpenML datasets = 60 tasks.

Examples:
  # Run all 60 clean real datasets in 2 parallel workers (4 threads each):
  python -m tools.bench_real_openml --out /tmp/bench60_clean --workers 2 --threads 4 --budget 45

  # Run 12 type-corrupted datasets (1 per narrow type) and compare against clean baseline:
  python -m tools.bench_real_openml --out /tmp/bench12_corrupt --tier1 --corrupt --workers 2 --threads 4 --budget 45 --baseline /tmp/bench60_clean/summary.json

  # Print scorecard from existing summary.json:
  python -m tools.bench_real_openml --scorecard /tmp/bench60_clean/summary.json
"""
import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]

# (group, name, openml_id, kind, metric, chrono, target_override, tier1_flag)
REGISTRY = [
    # 01. Binary standard / scoring
    ('01_binary_std', 'adult', 1590, 'binary', 'ROC-AUC', False, None, True),
    ('01_binary_std', 'credit_g', 31, 'binary', 'ROC-AUC', False, None, False),
    ('01_binary_std', 'australian', 40981, 'binary', 'ROC-AUC', False, None, False),
    ('01_binary_std', 'phishing', 4534, 'binary', 'ROC-AUC', False, None, False),
    ('01_binary_std', 'telco_churn', 42178, 'binary', 'LogLoss', False, 'Churn', False),
    # 02. Extreme imbalance / Fraud / Anomaly
    ('02_fraud_imbal', 'creditcard', 1597, 'binary', 'PR-AUC', False, None, False),
    ('02_fraud_imbal', 'aps_failure', 41138, 'binary', 'PR-AUC', False, None, False),
    ('02_fraud_imbal', 'mammography', 310, 'binary', 'PR-AUC', False, None, False),
    ('02_fraud_imbal', 'porto_seguro', 42206, 'binary', 'PR-AUC', False, None, False),
    ('02_fraud_imbal', 'give_me_credit', 46929, 'binary', 'PR-AUC', False, None, True),
    # 03. Multiclass small/medium (3-7 classes, imbalanced)
    ('03_multi_small', 'satimage', 182, 'multiclass', 'LogLoss', False, None, True),
    ('03_multi_small', 'covertype', 1596, 'multiclass', 'LogLoss', False, None, False),
    ('03_multi_small', 'shuttle', 40685, 'multiclass', 'F1-macro', False, None, False),
    ('03_multi_small', 'page_blocks', 30, 'multiclass', 'F1-macro', False, None, False),
    ('03_multi_small', 'diabetes130us', 4541, 'multiclass', 'LogLoss', False, None, False),
    # 04. Multiclass many (10-100 classes)
    ('04_multi_many', 'helena_100cls', 41169, 'multiclass', 'LogLoss', False, None, False),
    ('04_multi_many', 'letter_26cls', 6, 'multiclass', 'LogLoss', False, None, True),
    ('04_multi_many', 'soybean_19cls', 42, 'multiclass', 'F1-macro', False, None, False),
    ('04_multi_many', 'volkert_10cls', 41166, 'multiclass', 'LogLoss', False, None, False),
    ('04_multi_many', 'cardiotocography', 1466, 'multiclass', 'F1-macro', False, None, False),
    # 05. Ordinal / Ratings
    ('05_ordinal', 'wine_red', 40691, 'multiclass', 'QWK', False, None, False),
    ('05_ordinal', 'wine_white', 40498, 'multiclass', 'QWK', False, None, True),
    ('05_ordinal', 'eucalyptus', 188, 'multiclass', 'QWK', False, None, False),
    ('05_ordinal', 'abalone', 42726, 'regression', 'MAE', False, None, False),
    ('05_ordinal', 'student_por', 44967, 'regression', 'MAE', False, None, False),
    # 06. Regression standard / outliers / physics
    ('06_reg_std', 'california_housing', 43939, 'regression', 'MAE', False, None, True),
    ('06_reg_std', 'diamonds', 42225, 'regression', 'MAE', False, None, False),
    ('06_reg_std', 'miami_housing', 44983, 'regression', 'MAE', False, None, False),
    ('06_reg_std', 'superconductivity', 44964, 'regression', 'RMSE', False, None, False),
    ('06_reg_std', 'cpu_activity', 44978, 'regression', 'RMSE', False, None, False),
    # 07. Skewed / Zero-inflated regression
    ('07_reg_skew_zero', 'ames_housing', 42563, 'regression', 'RMSLE', False, None, True),
    ('07_reg_skew_zero', 'allstate_claims', 42571, 'regression', 'MAE', False, None, False),
    ('07_reg_skew_zero', 'fremtpl2freq', 41214, 'regression', 'MAE', False, 'ClaimNb', False),
    ('07_reg_skew_zero', 'brazilian_houses', 42688, 'regression', 'RMSLE', False, None, False),
    ('07_reg_skew_zero', 'medical_charges', 42720, 'regression', 'MAPE', False, None, False),
    # 08. Time-series regression (chronological split)
    ('08_ts_reg', 'bike_sharing', 42712, 'regression', 'MAE', True, None, True),
    ('08_ts_reg', 'nyc_taxi_tips', 42729, 'regression', 'MAE', True, None, False),
    ('08_ts_reg', 'ukair_pm10', 44068, 'regression', 'MAE', True, None, False),
    ('08_ts_reg', 'king_house_sales', 42731, 'regression', 'MAE', True, None, False),
    ('08_ts_reg', 'airlines_delay_reg', 42721, 'regression', 'MAE', True, None, False),
    # 09. Temporal classification / concept drift (chronological split)
    ('09_ts_cls', 'electricity', 151, 'binary', 'ROC-AUC', True, None, True),
    ('09_ts_cls', 'airlines_cls', 1169, 'binary', 'ROC-AUC', True, None, False),
    ('09_ts_cls', 'sf_police', 42732, 'binary', 'PR-AUC', True, None, False),
    ('09_ts_cls', 'med_noshow', 43439, 'binary', 'PR-AUC', True, 'No-show', False),
    ('09_ts_cls', 'bank_marketing', 1461, 'binary', 'PR-AUC', True, None, False),
    # 10. Tabular + Text (NLP + metadata)
    ('10_text_tab', 'kickstarter', 46810, 'binary', 'ROC-AUC', False, None, False),
    ('10_text_tab', 'fake_jobs', 46809, 'binary', 'PR-AUC', False, None, True),
    ('10_text_tab', 'mercari_price', 46886, 'regression', 'RMSE', False, None, False),
    ('10_text_tab', 'wine_reviews_30c', 46653, 'multiclass', 'LogLoss', False, None, False),
    ('10_text_tab', 'women_reviews', 46885, 'regression', 'MAE', False, None, False),
    # 11. High-cardinality ID / RecSys / CTR
    ('11_highcard_rec', 'amazon_access', 43900, 'binary', 'ROC-AUC', False, None, True),
    ('11_highcard_rec', 'click_pred', 42733, 'binary', 'PR-AUC', False, None, False),
    ('11_highcard_rec', 'black_friday', 41540, 'regression', 'MAE', False, None, False),
    ('11_highcard_rec', 'okcupid_stem', 42734, 'multiclass', 'LogLoss', False, None, False),
    ('11_highcard_rec', 'open_payments', 42738, 'binary', 'PR-AUC', False, None, False),
    # 12. Dirty / Wide / Sparse DWH tables
    ('12_dirty_wide', 'kick_auctions', 41162, 'binary', 'PR-AUC', False, None, True),
    ('12_dirty_wide', 'kdd98_478col', 42343, 'binary', 'PR-AUC', False, None, False),
    ('12_dirty_wide', 'speed_dating', 40536, 'binary', 'ROC-AUC', False, None, False),
    ('12_dirty_wide', 'road_safety', 42803, 'multiclass', 'LogLoss', False, None, False),
    ('12_dirty_wide', 'santander_4992c', 42572, 'regression', 'RMSLE', False, None, False),
]

REG_MAP = {r[1]: r for r in REGISTRY}
METRIC_CODE = {
    'PR-AUC': 'average_precision', 'ROC-AUC': 'roc_auc', 'LogLoss': 'logloss',
    'F1-macro': 'f1_macro', 'QWK': 'qwk', 'MAE': 'mae', 'RMSE': 'rmse',
    'RMSLE': 'rmsle', 'MAPE': 'mape',
}
HIGHER_IS_BETTER = {'PR-AUC', 'ROC-AUC', 'F1-macro', 'QWK'}
ORDINAL_ORDERS = {
    'eucalyptus': ['none', 'low', 'average', 'good', 'best'],
}


def _load_cached(item, cache_dir: Path, max_rows: int = 40_000):
    grp, name, did, kind, metric, chrono, tgt_ov, _ = item
    pq = cache_dir / f'{name}.parquet'
    mj = cache_dir / f'{name}.meta.json'
    if pq.exists() and mj.exists():
        return pd.read_parquet(pq), json.loads(mj.read_text(encoding='utf-8'))
    from sklearn.datasets import fetch_openml
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = fetch_openml(data_id=did, as_frame=True, parser='auto', data_home=str(cache_dir / '_sklearn_raw'))
    df = ds.frame.copy()
    tgt = tgt_ov if (tgt_ov and tgt_ov in df.columns) else (
        ds.target.name if (ds.target is not None and hasattr(ds.target, 'name') and ds.target.name in df.columns) else df.columns[-1]
    )
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True) if chrono else df.sample(max_rows, random_state=42).reset_index(drop=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
            df[c] = df[c].astype('string')
    df.to_parquet(pq, index=False)
    meta = {'group': grp, 'name': name, 'openml_id': did, 'kind': kind, 'metric': metric, 'chrono': chrono, 'target': str(tgt)}
    mj.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return df, meta


def _prepare_split(df: pd.DataFrame, meta: dict, seed: int = 42):
    from sklearn.model_selection import train_test_split
    kind = meta['kind']
    tgt = meta['target']
    name = meta['name']
    chrono = bool(meta['chrono'])

    df = df.copy().reset_index(drop=True)
    y_raw = df.pop(tgt)
    ok_y = y_raw.notna().to_numpy()
    if kind == 'regression':
        y_num = pd.to_numeric(y_raw, errors='coerce').to_numpy(dtype=float)
        ok_y = ok_y & np.isfinite(y_num)
        df = df.loc[ok_y].reset_index(drop=True)
        y_arr = y_num[ok_y]
        classes = []
    elif kind == 'binary':
        df = df.loc[ok_y].reset_index(drop=True)
        ys = y_raw.loc[ok_y].astype(str).str.strip().reset_index(drop=True)
        vc = ys.value_counts()
        pos_val = vc.index[-1]
        y_arr = (ys.to_numpy() == pos_val).astype(int)
        classes = [0, 1]
    else:
        df = df.loc[ok_y].reset_index(drop=True)
        ys = y_raw.loc[ok_y].astype(str).str.strip().reset_index(drop=True)
        if name in ORDINAL_ORDERS:
            ord_list = [v for v in ORDINAL_ORDERS[name] if v in set(ys.unique())]
            extra = sorted(set(ys.unique()) - set(ord_list))
            uniq = ord_list + extra
        else:
            u_vals = list(ys.unique())
            try:
                uniq = sorted(u_vals, key=lambda v: float(v))
            except Exception:
                uniq = sorted(u_vals)
        cmap = {v: i for i, v in enumerate(uniq)}
        y_arr = ys.map(cmap).to_numpy(dtype=int)
        classes = list(range(len(uniq)))

    # Ensure unique row_id column
    if 'row_id' in df.columns:
        df = df.rename(columns={'row_id': 'orig_row_id'})
    n = len(df)
    df.insert(0, 'row_id', np.arange(n, dtype=np.int64))

    if chrono:
        k = int(n * 0.8)
        tr_idx, te_idx = np.arange(k), np.arange(k, n)
        # Ensure all classes appear in train for classification
        if kind != 'regression' and len(np.unique(y_arr[tr_idx])) < 2:
            tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=y_arr)
    else:
        if kind != 'regression':
            vc_c = pd.Series(y_arr).value_counts()
            can_strat = int(vc_c.min()) >= 2
            tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=y_arr if can_strat else None)
            # Guarantee every class seen in test also exists in train
            tr_set = set(y_arr[tr_idx].tolist())
            miss_cls = [c for c in set(y_arr[te_idx].tolist()) if c not in tr_set]
            if miss_cls:
                move = [idx for idx in te_idx if y_arr[idx] in miss_cls]
                tr_idx = np.concatenate([tr_idx, move])
                te_idx = np.array([idx for idx in te_idx if idx not in set(move)])
        else:
            tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.2, random_state=seed)

    tr = df.iloc[tr_idx].copy().reset_index(drop=True)
    te = df.iloc[te_idx].copy().reset_index(drop=True)
    y_tr = y_arr[tr_idx].copy()
    y_te = y_arr[te_idx].copy()
    return tr, te, y_tr, y_te, classes


def corrupt_dataset(group: str, tr: pd.DataFrame, te: pd.DataFrame, y_tr: np.ndarray, y_te: np.ndarray,
                    kind: str, metric: str, classes: list, seed: int = 42):
    """Smart Type-Aware Chaos Corruptor: applies universal DWH dirt + type-specific Private LB trap."""
    rng = np.random.default_rng(seed + 777)
    tr, te = tr.copy(), te.copy()
    y_tr_out = pd.Series(y_tr, dtype=object if (group == '06_reg_std' and kind == 'regression') else None)
    aux, fks = {}, []
    notes = ['base_dwh_chaos']

    feat_cols = [c for c in tr.columns if c != 'row_id']
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(tr[c]) and not pd.api.types.is_bool_dtype(tr[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]

    # --- BASE DWH CHAOS (applied across all 12 narrow ML types) ---
    # 1) Dangerous headers (spaces, Cyrillic, LightGBM-breaking brackets)
    if len(feat_cols) >= 2:
        ren = {feat_cols[0]: f'признак [0]: {{{feat_cols[0]}}}'}
        tr = tr.rename(columns=ren); te = te.rename(columns=ren)
        feat_cols = [ren.get(c, c) for c in feat_cols]
        num_cols = [ren.get(c, c) for c in num_cols]
        cat_cols = [ren.get(c, c) for c in cat_cols]

    # 2) Convert 1 numeric col into dirty European strings ("12,5%", "N/A", "-")
    if num_cols:
        dc = num_cols[0]
        def _dirty_num(s):
            vals = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
            out = []
            for i, v in enumerate(vals):
                if not np.isfinite(v) or i % 29 == 0:
                    out.append('N/A' if i % 2 == 0 else '-')
                else:
                    out.append(f' {v:.3f}'.replace('.', ',') + ('%' if i % 5 == 0 else ''))
            return pd.Series(out, dtype='string')
        tr[dc] = _dirty_num(tr[dc]); te[dc] = _dirty_num(te[dc])
        num_cols = num_cols[1:]

    # 3) Missing values (20% random NaN on 1-2 cols) + all-NaN-in-test + constant-in-train + UUID junk + Postgres 't'/'f'
    for mc in num_cols[:2]:
        m_tr = rng.random(len(tr)) < 0.20
        m_te = rng.random(len(te)) < 0.20
        tr[mc] = pd.to_numeric(tr[mc], errors='coerce').astype(float)
        te[mc] = pd.to_numeric(te[mc], errors='coerce').astype(float)
        tr.loc[m_tr, mc] = np.nan; te.loc[m_te, mc] = np.nan
    tr['chaos__all_nan_test'] = rng.normal(size=len(tr)).astype('float32')
    te['chaos__all_nan_test'] = np.nan
    tr['chaos__const_train'] = 7.0
    te['chaos__const_train'] = rng.normal(size=len(te)).astype('float32')
    tr['chaos__uuid_junk'] = [f'UUID-{i:08x}-TR' for i in range(len(tr))]
    te['chaos__uuid_junk'] = [f'UUID-{i:08x}-TE' for i in range(len(te))]
    tr['chaos__pg_bool'] = pd.Series(np.where(rng.random(len(tr)) < 0.4, 't', 'f'), dtype='string')
    te['chaos__pg_bool'] = pd.Series(np.where(rng.random(len(te)) < 0.4, 't', 'f'), dtype='string')

    # 4) Extreme spikes & +-inf in 1 numeric column
    if len(num_cols) >= 2:
        sc = num_cols[1]
        tr[sc] = pd.to_numeric(tr[sc], errors='coerce').astype(float)
        te[sc] = pd.to_numeric(te[sc], errors='coerce').astype(float)
        s_tr = rng.random(len(tr)) < 0.005
        s_te = rng.random(len(te)) < 0.005
        tr.loc[s_tr, sc] = np.where(rng.random(int(s_tr.sum())) < 0.5, np.inf, -np.inf)
        te.loc[s_te, sc] = np.where(rng.random(int(s_te.sum())) < 0.5, 1e9, -np.inf)

    # 5) Pack 2 real numeric cols into flat JSONB + test-only secret key + Postgres {a,b} array tag
    if len(num_cols) >= 4:
        j1, j2 = num_cols[2], num_cols[3]
        v1a, v2a = pd.to_numeric(tr[j1], errors='coerce').fillna(0.0).to_numpy(), pd.to_numeric(tr[j2], errors='coerce').fillna(0.0).to_numpy()
        v1b, v2b = pd.to_numeric(te[j1], errors='coerce').fillna(0.0).to_numpy(), pd.to_numeric(te[j2], errors='coerce').fillna(0.0).to_numpy()
        tr['chaos__meta_json'] = [f'{{"k1": {a:.4f}, "k2": {b:.4f}, "grp": "{"hi" if a > 0 else "lo"}"}}' for a, b in zip(v1a, v2a)]
        te['chaos__meta_json'] = [f'{{"k1": {a:.4f}, "k2": {b:.4f}, "grp": "{"hi" if a > 0 else "lo"}", "secret_te": 999}}' for a, b in zip(v1b, v2b)]
        tr = tr.drop(columns=[j1, j2]); te = te.drop(columns=[j1, j2])
        num_cols = [c for c in num_cols if c not in (j1, j2)]
    tag_voc = np.array(['alpha', 'beta', 'gamma', 'delta', 'omega'])
    tr['chaos__pg_tags'] = ['{' + ','.join(rng.choice(tag_voc, size=int(rng.integers(1, 4)), replace=False)) + '}' for _ in range(len(tr))]
    te['chaos__pg_tags'] = ['{' + ','.join(rng.choice(tag_voc, size=int(rng.integers(1, 4)), replace=False)) + '}' for _ in range(len(te))]

    # --- TYPE-SPECIFIC PRIVATE-LB TRAPS (12 distinct profiles!) ---
    if group == '01_binary_std':
        notes.append('unseen_cats_15pct+dupes_10pct')
        if cat_cols:
            cc = cat_cols[0]
            te[cc] = te[cc].astype('string')
            te.loc[rng.random(len(te)) < 0.15, cc] = 'UNSEEN_CAT_XYZ'
        # Add 10% exact duplicate rows in train
        dup_idx = rng.choice(len(tr), size=max(10, len(tr) // 10), replace=False)
        tr = pd.concat([tr, tr.iloc[dup_idx]], ignore_index=True)
        y_tr_out = pd.concat([y_tr_out, y_tr_out.iloc[dup_idx]], ignore_index=True)

    elif group == '02_fraud_imbal':
        notes.append('non_id_sql_fk_dim+missing_1pct_fk')
        n_ent = 60
        e_tr = rng.integers(0, n_ent, size=len(tr))
        e_te = rng.integers(0, n_ent, size=len(te))
        tr['acct_ref'] = pd.Series([f'ENT_{e}' if i % 97 else None for i, e in enumerate(e_tr)], dtype='string')
        te['acct_ref'] = pd.Series([f'ENT_{e}' if i % 83 else 'ENT_UNSEEN' for i, e in enumerate(e_te)], dtype='string')
        aux['entity_dim'] = pd.DataFrame({
            'dim_code': [f'ENT_{i}' for i in range(n_ent)],
            'dim_risk': rng.normal(size=n_ent).astype('float32'),
            'dim_tier': rng.choice(['T1', 'T2', 'T3'], size=n_ent),
        })
        fks.append({'src_table': 'train', 'src_col': 'acct_ref', 'dst_table': 'entity_dim', 'dst_col': 'dim_code'})

    elif group == '03_multi_small':
        notes.append('label_shift_prior_reversal')
        # Reweight test classes to simulate genuine label shift
        vc = pd.Series(y_te).value_counts()
        inv_w = np.array([1.0 / max(vc.get(c, 1), 1) for c in y_te], dtype=float)
        inv_w /= inv_w.sum()
        sel_te = rng.choice(len(te), size=len(te), replace=True, p=inv_w)
        te = te.iloc[sel_te].reset_index(drop=True)
        te['row_id'] = np.arange(10_000_000, 10_000_000 + len(te), dtype=np.int64)
        y_te = y_te[sel_te].copy()

    elif group == '04_multi_many':
        notes.append('ultra_rare_class_n3+unseen_cats')
        # Keep only 3 train rows for one class
        rare_c = int(pd.Series(y_tr_out).value_counts().index[-1])
        c_pos = np.flatnonzero(y_tr_out.to_numpy() == rare_c)
        if len(c_pos) > 3:
            drop_p = set(c_pos[3:].tolist())
            keep_m = np.array([i not in drop_p for i in range(len(tr))])
            tr = tr.loc[keep_m].reset_index(drop=True)
            y_tr_out = y_tr_out.loc[keep_m].reset_index(drop=True)
        tr['chaos__cat_new'] = pd.Series(rng.choice(['A', 'B', 'C'], len(tr)), dtype='string')
        te['chaos__cat_new'] = pd.Series(rng.choice(['A', 'NEW_1', 'NEW_2'], len(te)), dtype='string')

    elif group == '05_ordinal':
        notes.append('messy_ordinal_strings+25pct_nan')
        for nc in num_cols[:2]:
            tr[nc] = pd.to_numeric(tr[nc], errors='coerce').astype(float)
            tr.loc[rng.random(len(tr)) < 0.25, nc] = np.nan

    elif group == '06_reg_std':
        notes.append('dirty_regression_target_1pct_na_comma')
        # Inject dirty target strings ("N/A", European decimal comma) in train target!
        vals = pd.to_numeric(y_tr_out, errors='coerce').to_numpy(dtype=float)
        dirty_y = []
        for i, v in enumerate(vals):
            if i % 101 == 0:
                dirty_y.append('N/A')
            elif i % 37 == 0:
                dirty_y.append(f'{v:.3f}'.replace('.', ','))
            else:
                dirty_y.append(f'{v:.4f}')
        y_tr_out = pd.Series(dirty_y, dtype='string')

    elif group == '07_reg_skew_zero':
        notes.append('zero_inflated_feats_30pct+5x_feat_spikes+0.5pct_y_outliers')
        for zc in num_cols[:3]:
            tr[zc] = pd.to_numeric(tr[zc], errors='coerce').astype(float)
            te[zc] = pd.to_numeric(te[zc], errors='coerce').astype(float)
            tr.loc[rng.random(len(tr)) < 0.30, zc] = 0.0
            te.loc[rng.random(len(te)) < 0.30, zc] = 0.0
            s_tr = rng.random(len(tr)) < 0.01
            s_te = rng.random(len(te)) < 0.01
            tr.loc[s_tr, zc] = tr.loc[s_tr, zc] * 5.0
            te.loc[s_te, zc] = te.loc[s_te, zc] * 5.0
        yv = pd.to_numeric(y_tr_out, errors='coerce').to_numpy(dtype=float)
        sp_m = rng.random(len(yv)) < 0.005
        yv[sp_m] = yv[sp_m] * 2.5
        y_tr_out = pd.Series(yv)

    elif group == '08_ts_reg':
        notes.append('unix_epoch_bigint+outage_block')
        base_ep = 1700000000
        tr['event_epoch_ts'] = (base_ep + np.arange(len(tr), dtype=np.int64) * 3600)
        te['event_epoch_ts'] = (base_ep + (len(tr) + np.arange(len(te), dtype=np.int64)) * 3600)
        # Simultaneous sensor outage block on 10% of rows
        if len(num_cols) >= 3:
            out_tr = rng.random(len(tr)) < 0.10
            out_te = rng.random(len(te)) < 0.10
            for nc in num_cols[:3]:
                tr[nc] = pd.to_numeric(tr[nc], errors='coerce').astype(float)
                te[nc] = pd.to_numeric(te[nc], errors='coerce').astype(float)
                tr.loc[out_tr, nc] = np.nan; te.loc[out_te, nc] = np.nan

    elif group == '09_ts_cls':
        notes.append('dmy_dot_dates+feature_dropout_drift')
        stamps = pd.date_range('2023-01-01', periods=len(tr) + len(te), freq='30min')
        tr['event_date'] = pd.Series(stamps[:len(tr)].strftime('%d.%m.%Y %H:%M:%S'), dtype='string')
        te['event_date'] = pd.Series(stamps[len(tr):].strftime('%d.%m.%Y %H:%M:%S'), dtype='string')
        if num_cols:
            dc = num_cols[0]
            te[dc] = pd.to_numeric(te[dc], errors='coerce').astype(float)
            te.iloc[len(te) // 2:, te.columns.get_loc(dc)] = np.nan

    elif group == '10_text_tab':
        notes.append('dirty_html_cyrillic_empty_text')
        txt_c = [c for c in cat_cols if tr[c].dropna().astype(str).str.len().mean() > 15]
        for tc in txt_c[:2]:
            tr[tc] = tr[tc].astype('string')
            te[tc] = te[tc].astype('string')
            m1 = rng.random(len(tr)) < 0.08
            m2 = rng.random(len(te)) < 0.08
            tr.loc[m1, tc] = '<b>N/A</b> пропущено '
            te.loc[m2, tc] = ''

    elif group == '11_highcard_rec':
        notes.append('cold_start_20pct_ids+2hop_snowflake_fk')
        if feat_cols:
            ic = feat_cols[0]
            te[ic] = te[ic].astype('string')
            te.loc[rng.random(len(te)) < 0.20, ic] = 'COLD_START_999999'
        # Add 2-hop snowflake SQL FK: train -> seg_mid -> seg_top
        n_m, n_t = 40, 10
        m2t = rng.integers(0, n_t, size=n_m)
        tr['mid_ref'] = pd.Series([f'M{i}' for i in rng.integers(0, n_m, len(tr))], dtype='string')
        te['mid_ref'] = pd.Series([f'M{i}' for i in rng.integers(0, n_m, len(te))], dtype='string')
        aux['seg_mid'] = pd.DataFrame({'mid_code': [f'M{i}' for i in range(n_m)], 'top_ref': [f'T{m2t[i]}' for i in range(n_m)], 'mid_val': rng.normal(size=n_m).astype('float32')})
        aux['seg_top'] = pd.DataFrame({'top_code': [f'T{i}' for i in range(n_t)], 'top_val': rng.normal(size=n_t).astype('float32')})
        fks.extend([
            {'src_table': 'train', 'src_col': 'mid_ref', 'dst_table': 'seg_mid', 'dst_col': 'mid_code'},
            {'src_table': 'seg_mid', 'src_col': 'top_ref', 'dst_table': 'seg_top', 'dst_col': 'top_code'},
        ])

    elif group == '12_dirty_wide':
        notes.append('plus50_junk_cols+mixed_obj_strings')
        junk_tr, junk_te = {}, {}
        for k_j in range(50):
            if k_j < 20:
                a = rng.normal(size=len(tr)).astype('float32'); b = rng.normal(size=len(te)).astype('float32')
                a[rng.random(len(a)) < 0.98] = np.nan; b[rng.random(len(b)) < 0.98] = np.nan
            elif k_j < 35:
                a = np.full(len(tr), float(k_j), dtype='float32'); b = np.full(len(te), float(k_j), dtype='float32')
            else:
                a = pd.Series(np.where(rng.random(len(tr)) < 0.3, 'unknown', '12.34'), dtype='string')
                b = pd.Series(np.where(rng.random(len(te)) < 0.3, 'unknown', '12.34'), dtype='string')
            junk_tr[f'junk_{k_j}'] = a; junk_te[f'junk_{k_j}'] = b
        tr = pd.concat([tr, pd.DataFrame(junk_tr)], axis=1)
        te = pd.concat([te, pd.DataFrame(junk_te)], axis=1)

    return tr, te, y_tr_out, y_te, aux, fks, notes


def _eval_metric(metric: str, kind: str, y_true: np.ndarray, pred_arr: np.ndarray, classes: list):
    from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, log_loss, mean_absolute_error, roc_auc_score
    if kind == 'binary':
        p1 = np.clip(pred_arr[:, 0], 0.0, 1.0)
        if metric == 'PR-AUC':
            return float(average_precision_score(y_true, p1))
        if metric == 'ROC-AUC':
            return float(roc_auc_score(y_true, p1))
        if metric == 'LogLoss':
            return float(log_loss(y_true, np.column_stack([1.0 - p1, p1]), labels=[0, 1]))
    elif kind == 'multiclass':
        if metric == 'QWK':
            lbl = np.rint(pred_arr[:, 0]).astype(int)
            return float(cohen_kappa_score(y_true, lbl, weights='quadratic'))
        p = np.clip(pred_arr, 1e-15, 1.0)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-15)
        if metric == 'LogLoss':
            return float(log_loss(y_true, p, labels=classes))
        if metric == 'F1-macro':
            return float(f1_score(y_true, p.argmax(axis=1), labels=classes, average='macro', zero_division=0))
    else:
        p = pred_arr[:, 0]
        if metric == 'MAE':
            return float(mean_absolute_error(y_true, p))
        if metric == 'RMSE':
            return float(np.sqrt(np.mean((y_true - p) ** 2)))
        if metric == 'RMSLE':
            return float(np.sqrt(np.mean((np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))
        if metric == 'MAPE':
            return float(np.mean(np.abs(y_true - p) / np.maximum(np.abs(y_true), 1e-9)))
    raise ValueError(f'Unsupported {kind}/{metric}')


def _naive_baseline(metric: str, kind: str, y_tr_clean: np.ndarray, y_te: np.ndarray, classes: list):
    n_te = len(y_te)
    if kind == 'binary':
        p = float(np.mean(y_tr_clean))
        return _eval_metric(metric, kind, y_te, np.full((n_te, 1), p), classes)
    if kind == 'multiclass':
        if metric == 'QWK':
            return 0.0
        cnt = np.bincount(y_tr_clean.astype(int), minlength=len(classes)).astype(float)
        pr = np.clip(cnt / max(cnt.sum(), 1.0), 1e-6, 1.0)
        pr /= pr.sum()
        return _eval_metric(metric, kind, y_te, np.tile(pr, (n_te, 1)), classes)
    y_ok = y_tr_clean[np.isfinite(y_tr_clean)]
    if metric == 'MAE':
        c = float(np.median(y_ok))
    elif metric == 'RMSLE':
        c = float(np.expm1(np.mean(np.log1p(np.clip(y_ok, 0, None)))))
    elif metric == 'MAPE':
        pos = y_ok[y_ok > 0]
        c = float(np.median(pos)) if len(pos) else 1.0
    else:
        c = float(np.mean(y_ok))
    return _eval_metric(metric, kind, y_te, np.full((n_te, 1), c), classes)


def _norm_100(metric: str, val: float, ref: float, clip_zero: bool = True):
    if val is None or ref is None or not (np.isfinite(val) and np.isfinite(ref)):
        return None
    if metric in HIGHER_IS_BETTER:
        s = 100.0 * (val - ref) / max(1.0 - ref, 1e-9)
    else:
        s = 100.0 * (1.0 - val / max(abs(ref), 1e-9))
    return float(max(0.0, s)) if clip_zero else float(s)


def worker(args):
    if hasattr(os, 'sched_setaffinity') and args.cpu_offset is not None:
        cpus = sorted(os.sched_getaffinity(0))
        sel = cpus[args.cpu_offset: args.cpu_offset + args.threads]
        if sel:
            os.sched_setaffinity(0, sel)
    from unittest.mock import patch
    import ag_database
    import solution

    item = REG_MAP[args.dataset]
    grp, name, did, kind, metric, chrono, _, _ = item
    df, meta = _load_cached(item, args.cache)
    tr, te, y_tr, y_te, classes = _prepare_split(df, meta, seed=args.seed)
    y_tr_clean = y_tr.copy()
    aux, fks, c_notes = {}, [], []
    if args.corrupt:
        tr, te, y_tr_ser, y_te, aux, fks, c_notes = corrupt_dataset(grp, tr, te, y_tr, y_te, kind, metric, classes, seed=args.seed)
        tr['target'] = y_tr_ser.values
    else:
        tr['target'] = y_tr

    rng = np.random.default_rng(args.seed + 13)
    sample = pd.DataFrame({'row_id': rng.permutation(te['row_id'].to_numpy())})
    if kind == 'multiclass' and metric != 'QWK':
        for ci in classes:
            sample[f'class_{ci}'] = 0.0
        t_desc = f'Multiclass classification; probabilities class_0 through class_{len(classes) - 1}'
    elif kind == 'multiclass' and metric == 'QWK':
        sample['target'] = 0
        t_desc = f'Multiclass ordinal classification; integer class label from 0 to {len(classes) - 1}'
    elif kind == 'binary':
        sample['target'] = 0.0
        t_desc = 'Binary classification; probability of class 1'
    else:
        sample['target'] = 0.0
        t_desc = 'Regression'

    d = args.out
    (d / 'task.md').write_text(
        f'ID: row_id. Target: target. {t_desc}. Основная метрика: {metric}.\n', encoding='utf-8'
    )
    naive = _naive_baseline(metric, kind, y_tr_clean, y_te, classes)
    chk = {
        'group': grp, 'dataset': name, 'openml_id': did, 'kind': kind, 'metric': metric,
        'chrono': chrono, 'corrupt': bool(args.corrupt), 'corrupt_notes': c_notes,
        'train_rows': len(tr), 'test_rows': len(te), 'cols': tr.shape[1] - 2,
        'n_classes': len(classes), 'naive_baseline': float(naive), 'issues': [],
    }
    # Optional temporal poison check on corrupted chronological datasets
    if args.corrupt and chrono:
        from ag_features import detect_time_split, prepare
        from ag_task import build_spec
        sp = build_spec(d / 'task.md', tr, te, sample, None, {})
        p1 = prepare(tr, te, sp)
        tc = detect_time_split(p1.X, p1.Xt, p1.cat_cols)
        chk['time_col_detected'] = tc
        if tc is None:
            chk['issues'].append('temporal split not detected under corrupt dates')
        else:
            te_p = te.copy()
            mid = len(te_p) // 2
            for col in te_p.columns:
                if col not in ('row_id', 'event_epoch_ts', 'event_date') and pd.api.types.is_numeric_dtype(te_p[col]):
                    te_p.iloc[mid:, te_p.columns.get_loc(col)] = 99999.0
            p2 = prepare(tr, te_p, sp)
            common_num = [c for c in p1.Xt.columns if c in p2.Xt.columns and pd.api.types.is_numeric_dtype(p1.Xt[c])]
            if common_num:
                a1 = p1.Xt.iloc[:mid][common_num].to_numpy(dtype=float)
                a2 = p2.Xt.iloc[:mid][common_num].to_numpy(dtype=float)
                diff = np.abs(a1 - a2)
                fin = np.isfinite(a1) & np.isfinite(a2)
                delta = float(np.max(diff[fin])) if fin.any() else 0.0
                chk['future_poison_delta'] = delta
                if delta > 1e-5:
                    chk['issues'].append(f'temporal poison delta={delta:.3g}')
        del p1
        gc.collect()

    eval_meta = d.parent / f'.eval_{d.name}.json'
    eval_meta.write_text(json.dumps({
        'chk': chk, 'classes': classes,
        'te_ids': te['row_id'].tolist(), 'y_te': y_te.tolist(),
        'sample_ids': sample['row_id'].tolist(), 'sample_cols': list(sample.columns),
    }), encoding='utf-8')

    os.environ['PGDATABASE'] = f'openml_{name}'
    def _mock_extract(out_dir, meta=None, **kw):
        if isinstance(meta, dict) and fks:
            meta['foreign_keys'] = list(fks)
        return {'train': tr, 'test': te, 'sample_submission': sample, **aux}

    with patch.object(ag_database, 'extract', side_effect=_mock_extract):
        solution.run(d / 'task.md')


def run_one(args, item, slot: int = 0):
    grp, name, did, kind, metric, _, _, _ = item
    tag = f'{name}_corrupt' if args.corrupt else name
    d = args.out / tag
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PG', 'OPENAI_', 'AGENT_'))}
    env.update(
        AGENT_TIME_BUDGET=str(args.budget),
        AGENT_HARD_LIMIT=str(args.budget + 45),
        PYTHONPATH=str(ROOT),
    )
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[var] = str(args.threads)
    cmd = [
        sys.executable, '-m', 'tools.bench_real_openml', '--worker',
        '--dataset', name, '--out', str(d), '--cache', str(args.cache),
        '--seed', str(args.seed), '--threads', str(args.threads),
        '--cpu-offset', str(slot * args.threads),
    ]
    if args.corrupt:
        cmd.append('--corrupt')
    t0 = time.monotonic()
    peak = 0
    timed_out = False
    with (d / 'stdout.log').open('w', encoding='utf-8') as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            rss = 0
            try:
                for ch in [p] + p.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            peak = max(peak, rss)
            if time.monotonic() - t0 > args.budget + args.grace:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                break
            time.sleep(0.15)
        rc = proc.wait()

    res = {
        'group': grp, 'dataset': name, 'openml_id': did, 'kind': kind, 'metric': metric,
        'corrupt': bool(args.corrupt), 'seconds': round(time.monotonic() - t0, 2),
        'peak_rss_gib': round(peak / 2**30, 3), 'exit_code': rc, 'timeout': timed_out, 'issues': [],
    }
    try:
        eval_meta = d.parent / f'.eval_{d.name}.json'
        me = json.loads(eval_meta.read_text(encoding='utf-8'))
        res.update(me['chk'])
        sub = pd.read_csv(d / 'submission.csv')
        if list(sub.columns) != me['sample_cols'] or len(sub) != len(me['sample_ids']) or not np.array_equal(sub['row_id'].to_numpy(), np.asarray(me['sample_ids'])):
            res['issues'].append('schema/ID order mismatch')
        pred_arr = sub.drop(columns=['row_id']).to_numpy(dtype=float)
        if not np.isfinite(pred_arr).all():
            res['issues'].append('NaN/Inf predictions')
        if kind == 'binary' and ((pred_arr < -1e-6).any() or (pred_arr > 1.0 + 1e-6).any()):
            res['issues'].append('binary proba out of [0,1]')
        if kind == 'multiclass' and metric != 'QWK' and not np.allclose(pred_arr.sum(1), 1.0, atol=1e-4):
            res['issues'].append('multiclass proba sum != 1')
        y_al = pd.Series(me['y_te'], index=me['te_ids']).loc[sub['row_id']].to_numpy(dtype=float)
        val = _eval_metric(metric, kind, y_al, pred_arr, me['classes'])
        res['score'] = float(val)
        res['score_100'] = round(_norm_100(metric, val, res['naive_baseline'], clip_zero=True) or 0.0, 2)
        res['constant'] = bool(np.all(np.ptp(pred_arr, axis=0) < 1e-10))
        if res['constant'] and metric != 'MAE':
            res['issues'].append('constant predictions')
        sp_p = d / 'agent_work' / 'spec.json'
        if sp_p.exists():
            sp_j = json.loads(sp_p.read_text(encoding='utf-8'))
            if sp_j.get('metric') != METRIC_CODE[metric]:
                res['issues'].append(f"metric mismatch: {sp_j.get('metric')} != {METRIC_CODE[metric]}")
        fin_p = d / 'agent_work' / 'final.json'
        if fin_p.exists():
            fj = json.loads(fin_p.read_text(encoding='utf-8'))
            res['oof'] = fj.get('oof_score')
            res['choice'] = fj.get('choice')
        else:
            res['issues'].append('final.json absent')
    except Exception as exc:
        res['issues'].append(f'{type(exc).__name__}: {exc}')

    if rc or timed_out:
        res['issues'].append('nonzero exit or timeout')
    if peak >= 24 * 2**30:
        res['issues'].append('peak RSS >= 24 GiB')
    log_txt = (d / 'stdout.log').read_text(errors='replace') if (d / 'stdout.log').exists() else ''
    if 'WATCHDOG:' in log_txt:
        res['issues'].append('watchdog triggered')
    errs = [ln for ln in log_txt.splitlines() if 'Traceback (most recent call last)' in ln or 'ОШИБКА' in ln][:10]
    if errs:
        res['logged_errors'] = errs
        res['issues'].append(f'{len(errs)} logged errors')
    res['status'] = 'FAIL' if res['issues'] else 'PASS'
    (d / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    sc_str = f"{res['score']:.4g}" if 'score' in res else 'N/A'
    print(f"[{res['status']:4s}] {grp:16s} {name:20s} ({metric:8s}) = {sc_str:>10s} | S={res.get('score_100', 0.0):6.2f}/100 | {res['seconds']:5.1f}s | {res['peak_rss_gib']:.2f}G" + (f" | ISSUES={res['issues']}" if res['issues'] else ''), flush=True)
    return res


def build_scorecard(results, baseline_results=None):
    base_map = {}
    if baseline_results:
        for r in baseline_results:
            if r.get('score') is not None:
                base_map[r['dataset']] = float(r['score'])
    by_group = {}
    s_all, deltas = [], []
    passed = 0
    for r in results:
        ok = r.get('status') == 'PASS'
        if ok:
            passed += 1
        s100 = float(r.get('score_100', 0.0)) if ok else 0.0
        s_all.append(s100)
        b_val = base_map.get(r['dataset'])
        d_pct = _norm_100(r['metric'], r.get('score'), b_val, clip_zero=False) if (ok and b_val is not None) else None
        if d_pct is not None:
            deltas.append(d_pct)
        by_group.setdefault(r['group'], []).append({**r, 'score_100': s100, 'delta_vs_base': round(d_pct, 2) if d_pct is not None else None})
    grp_summary = {}
    for g in sorted(by_group):
        rows = by_group[g]
        g_pass = sum(1 for x in rows if x['status'] == 'PASS')
        g_s = [x['score_100'] for x in rows]
        g_d = [x['delta_vs_base'] for x in rows if x['delta_vs_base'] is not None]
        grp_summary[g] = {
            'passed': g_pass, 'total': len(rows),
            'mean_S': round(float(np.mean(g_s)), 2) if g_s else 0.0,
            'min_S': round(float(np.min(g_s)), 2) if g_s else 0.0,
            'mean_delta': round(float(np.mean(g_d)), 2) if g_d else None,
        }
    gate_ok = bool(results) and passed == len(results)
    return {
        'gate': 'PASS' if gate_ok else 'BLOCKED',
        'passed': passed, 'total': len(results),
        'composite_score': round(float(np.mean(s_all)), 2) if (gate_ok and s_all) else 0.0,
        'raw_mean_score': round(float(np.mean(s_all)), 2) if s_all else 0.0,
        'worst_case_score': round(float(np.min(s_all)), 2) if s_all else 0.0,
        'mean_delta_pct': round(float(np.mean(deltas)), 2) if deltas else None,
        'worst_delta_pct': round(float(np.min(deltas)), 2) if deltas else None,
        'groups': grp_summary,
        'failed_datasets': [{'dataset': r['dataset'], 'group': r['group'], 'issues': r.get('issues')} for r in results if r.get('status') != 'PASS'],
    }


def print_scorecard(sc):
    print('\n=== REAL OPENML RELEASE SCORECARD (12 NARROW ML TYPES) ===')
    for g, gs in sc['groups'].items():
        d_s = f" | d={gs['mean_delta']:+6.2f}%" if gs['mean_delta'] is not None else ''
        print(f"  {g:18s} : {gs['passed']:2d}/{gs['total']:<2d} PASS | mean S={gs['mean_S']:6.2f}/100 | min S={gs['min_S']:6.2f}{d_s}")
    d_tot = f" | mean d={sc['mean_delta_pct']:+.2f}% | worst d={sc['worst_delta_pct']:+.2f}%" if sc['mean_delta_pct'] is not None else ''
    print(f"GATE={sc['gate']} ({sc['passed']}/{sc['total']}) | Composite={sc['composite_score']:.2f}/100 (raw_mean={sc['raw_mean_score']:.2f}) | Worst={sc['worst_case_score']:.2f}{d_tot}", flush=True)
    if sc['failed_datasets']:
        print('FAILURES:')
        for f in sc['failed_datasets']:
            print(f"  - {f['group']} / {f['dataset']}: {f['issues']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('/tmp/bench_real_openml'))
    p.add_argument('--cache', type=Path, default=Path('/tmp/openml_cache_v12'))
    p.add_argument('--budget', type=float, default=45)
    p.add_argument('--grace', type=float, default=75)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tier1', action='store_true', help='run 12 representative datasets (1 per narrow ML type)')
    p.add_argument('--corrupt', action='store_true', help='enable Smart Type-Aware Chaos Corruptor')
    p.add_argument('--dataset', help='single dataset name from REGISTRY')
    p.add_argument('--only', help='comma-separated dataset names from REGISTRY')
    p.add_argument('--group', help='single group name from REGISTRY')
    p.add_argument('--merge', type=Path, help='merge newly run results into an existing summary.json')
    p.add_argument('--baseline', type=Path, help='baseline summary.json for delta comparison')
    p.add_argument('--scorecard', type=Path, help='print scorecard from existing summary.json')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--cpu-offset', type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.scorecard:
        res = json.loads(a.scorecard.read_text(encoding='utf-8'))
        base = json.loads(a.baseline.read_text(encoding='utf-8')) if a.baseline and a.baseline.exists() else None
        sc = build_scorecard(res, base)
        print_scorecard(sc)
        sys.exit(0 if sc['gate'] == 'PASS' else 1)

    if a.worker:
        worker(a)
        return

    a.out.mkdir(parents=True, exist_ok=True)
    items = list(REGISTRY)
    if a.dataset:
        items = [REG_MAP[a.dataset]]
    elif a.only:
        want = [x.strip() for x in a.only.split(',') if x.strip()]
        items = [REG_MAP[w] for w in want]
    elif a.group:
        items = [r for r in items if r[0] == a.group]
    elif a.tier1:
        items = [r for r in items if r[7]]

    prev_merged = json.loads(a.merge.read_text(encoding='utf-8')) if (a.merge and a.merge.exists()) else []
    results = []
    t_all = time.monotonic()
    if a.workers <= 1 or len(items) == 1:
        for it in items:
            results.append(run_one(a, it, slot=0))
            (a.out / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        import queue
        slots = queue.Queue()
        for s in range(a.workers):
            slots.put(s)

        def _wrapped(it):
            sl = slots.get()
            try:
                return run_one(a, it, slot=sl)
            finally:
                slots.put(sl)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_wrapped, it) for it in items]
            for fut in as_completed(futs):
                results.append(fut.result())
                results_sorted = sorted(results, key=lambda r: (r['group'], r['dataset']))
                (a.out / 'summary.json').write_text(json.dumps(results_sorted, ensure_ascii=False, indent=2), encoding='utf-8')
        results = sorted(results, key=lambda r: (r['group'], r['dataset']))

    if prev_merged:
        by_key = {(r['dataset'], bool(r.get('corrupt', False))): r for r in prev_merged}
        for r in results:
            by_key[(r['dataset'], bool(r.get('corrupt', False)))] = r
        results = sorted(by_key.values(), key=lambda r: (r['group'], r['dataset']))
        (a.out / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    base = json.loads(a.baseline.read_text(encoding='utf-8')) if a.baseline and a.baseline.exists() else None
    sc = build_scorecard(results, base)
    sc['total_wall_seconds'] = round(time.monotonic() - t_all, 1)
    (a.out / 'scorecard.json').write_text(json.dumps(sc, ensure_ascii=False, indent=2), encoding='utf-8')
    print_scorecard(sc)
    sys.exit(0 if sc['gate'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
