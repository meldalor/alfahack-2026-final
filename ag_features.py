"""Подготовка признаков и библиотека преобразований, из которых строятся гипотезы."""

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ag_utils import log, log_exc

DATE_RE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?")
DMY_DOT_RE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")
_NA_TOKS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "unknown", "missing", "inf", "-inf"}


def _parse_dirty_numeric(s: pd.Series):
    """Parse numeric strings with whitespace, NA tokens, trailing '%', or European decimal comma ('12,5')."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        return num, int(num.notna().sum())
    st = s.astype("string").str.strip()
    clean = st.mask(st.str.lower().isin(_NA_TOKS))
    num = pd.to_numeric(clean, errors="coerce")
    nn_c = int(clean.notna().sum())
    if nn_c > 0 and num.notna().sum() >= 0.95 * nn_c:
        return num, nn_c
    alt = clean.str.replace(r"%$", "", regex=True).str.strip()
    if nn_c > 0 and alt.dropna().str.match(r"^-?\d+,\d+$").mean() > 0.5:
        alt = alt.str.replace(",", ".", regex=False)
    num2 = pd.to_numeric(alt, errors="coerce")
    return num2, nn_c


def _unix_epoch_unit(name: str, a: pd.Series) -> str | None:
    low = str(name).lower()
    if not (low in ("timestamp", "ts", "dt", "time", "date", "epoch")
            or low.endswith(("_time", "_timestamp", "_date", "_datetime", "_at", "_ts", "_dt", "_epoch"))
            or low.startswith(("timestamp_", "datetime_", "event_time", "unix_"))):
        return None
    if not pd.api.types.is_numeric_dtype(a) or pd.api.types.is_bool_dtype(a):
        return None
    s = pd.to_numeric(a, errors="coerce").dropna()
    if len(s) < 5 or s.nunique() < 3:
        return None
    vals = s.to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return None
    lo, hi = float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))
    if 9e8 <= lo and hi <= 2.6e9:
        return "s"
    if 9e11 <= lo and hi <= 2.6e12:
        return "ms"
    return None


@dataclass
class Prepared:
    X: pd.DataFrame
    Xt: pd.DataFrame
    y: np.ndarray
    cat_cols: list
    text_cols: list = field(default_factory=list)
    raw_text: dict = field(default_factory=dict)
    time_col: str | None = None
    info: dict = field(default_factory=dict)


def _is_datetime_series(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(300)
    if len(sample) < 5:
        return False
    return sample.map(lambda v: bool(DATE_RE.match(v))).mean() > 0.9


def _codes_seen(name: str, a: pd.Series, b: pd.Series) -> bool:
    """Код-категория осмысленна, только если значения теста в основном есть в train. Иначе это порядковая колонка
    (год, номер периода): новые «категории» бесполезны и прячут временной сплит."""
    b = b.dropna()
    if len(b) and float(b.isin(set(a.dropna().unique())).mean()) < 0.5:
        log(f"{name}: значений теста нет в train — колонка остаётся числовой, а не категорией")
        return False
    return True


def _expand_datetime(name: str, a: pd.Series, b: pd.Series):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pd.api.types.is_datetime64_any_dtype(a):
            pa = pd.to_datetime(a, errors="coerce", utc=True)
            pb = pd.to_datetime(b, errors="coerce", utc=True)
        else:
            sa_smp = a.dropna().astype(str).head(100)
            dayfirst = bool(len(sa_smp) and sa_smp.map(lambda v: bool(DMY_DOT_RE.match(v))).mean() > 0.7)
            pa = pd.to_datetime(a, errors="coerce", utc=True, dayfirst=dayfirst)
            pb = pd.to_datetime(b, errors="coerce", utc=True, dayfirst=dayfirst)
            if pa.notna().mean() < 0.5 and a.notna().any():
                pa = pd.to_datetime(a, errors="coerce", format="mixed", utc=True, dayfirst=dayfirst)
                pb = pd.to_datetime(b, errors="coerce", format="mixed", utc=True, dayfirst=dayfirst)
    both = pd.concat([pa, pb], ignore_index=True)
    out = pd.DataFrame(index=both.index)
    out[f"{name}__year"] = both.dt.year
    out[f"{name}__month"] = both.dt.month
    out[f"{name}__day"] = both.dt.day
    out[f"{name}__dow"] = both.dt.dayofweek
    out[f"{name}__doy"] = both.dt.dayofyear
    if (both.dt.hour.fillna(0) != 0).mean() > 0.05:
        out[f"{name}__hour"] = both.dt.hour
    if (both.dt.minute.fillna(0) != 0).mean() > 0.05:
        out[f"{name}__minute"] = both.dt.minute
    # Origin strictly from train min timestamp: keeps float32 second-level precision and zero test dependence.
    origin = pa.min()
    if pd.isna(origin):
        origin = pb.min()
    if pd.isna(origin):
        origin = pd.Timestamp("1970-01-01", tz="UTC")
    out[f"{name}__ord"] = (both - origin).dt.total_seconds() / 86400.0
    for col in out.columns:
        out[col] = out[col].astype("float64" if col.endswith("__ord") else "float32")
    a_out = out.iloc[: len(a)].reset_index(drop=True)
    b_out = out.iloc[len(a):].reset_index(drop=True)
    a_out.attrs["origin"] = origin
    return a_out, b_out


def _unpack_flat_json(name: str, sa: pd.Series, sb: pd.Series, max_keys: int = 8):
    """Unpack flat JSON/JSONB object strings using strictly train-observed scalar keys (>20% valid rows, <=8 keys)."""
    sample = sa.dropna().str.strip().head(500)
    if len(sample) < 10:
        return None
    looks_json = (sample.str.startswith("{") & sample.str.endswith("}") & sample.str.contains(":", regex=False)).mean()
    if looks_json < 0.20:
        return None
    key_counts = {}
    n_valid = 0
    for raw in sample.tolist():
        if not raw or raw[0] != "{" or raw[-1] != "}":
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        n_valid += 1
        for k, v in obj.items():
            if v is None or isinstance(v, (bool, int, float, str)):
                ks = str(k)
                key_counts[ks] = key_counts.get(ks, 0) + 1
    if n_valid < max(5, int(0.20 * len(sample))) or not key_counts:
        return None
    min_cnt = max(2, int(0.10 * n_valid))
    picked = sorted(((-cnt, k) for k, cnt in key_counts.items() if cnt >= min_cnt))[:max_keys]
    keys = [k for _, k in picked]
    if not keys:
        return None

    def _extract(series: pd.Series):
        cols_out = {k: [None] * len(series) for k in keys}
        nkeys = np.zeros(len(series), dtype=np.float32)
        for i, val in enumerate(series.tolist()):
            if val is None or val is pd.NA:
                continue
            sv = str(val).strip()
            if not sv or sv[0] != "{" or sv[-1] != "}":
                continue
            try:
                obj = json.loads(sv)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            nkeys[i] = float(len(obj))
            for k in keys:
                v = obj.get(k)
                if isinstance(v, bool):
                    cols_out[k][i] = 1.0 if v else 0.0
                elif isinstance(v, (int, float, str)):
                    cols_out[k][i] = v
        return cols_out, nkeys

    a_map, a_nk = _extract(sa)
    b_map, b_nk = _extract(sb)
    a_df, b_df = {f"{name}__json_nkeys": a_nk}, {f"{name}__json_nkeys": b_nk}
    new_cats = []
    for k in keys:
        col_nm = f"{name}__json__{k}"
        s_a = pd.Series(a_map[k], dtype=object)
        s_b = pd.Series(b_map[k], dtype=object)
        num_a, nn_a = _parse_dirty_numeric(s_a)
        if nn_a > 0 and num_a.notna().sum() >= 0.90 * nn_a:
            num_b, _ = _parse_dirty_numeric(s_b)
            a_df[col_nm] = num_a.astype("float32").to_numpy()
            b_df[col_nm] = num_b.astype("float32").to_numpy()
        else:
            st_a = s_a.astype("string")
            nu = int(st_a.nunique(dropna=True))
            if 2 <= nu <= 500:
                a_df[col_nm] = st_a
                b_df[col_nm] = s_b.astype("string")
                new_cats.append(col_nm)
    return pd.DataFrame(a_df), pd.DataFrame(b_df), new_cats


def _add_spectral_graph_features(X: pd.DataFrame, Xt: pd.DataFrame, cat_cols: list):
    """Zero-GNN 2-fold OOF Procrustes-aligned spectral bipartite embeddings (PPMI+SVD) strictly on train rows.
    Activated only on non-temporal dense community graphs (avg_deg>=3, spectral gap>=1.85); zero self-edge bias."""
    from scipy import sparse
    from scipy.sparse.linalg import svds

    n_tr, n_te = len(X), len(Xt)
    if n_tr < 200 or n_tr > 200_000 or X.shape[1] >= 900:
        return X, Xt
    g_cols = []
    seen_gc = set()
    # Include both categorical entity columns and raw integer *_id columns (e.g. user_id, merchant_id)
    cand_pool = list(cat_cols) + [
        c for c in X.columns
        if c not in cat_cols and str(c).lower().endswith("_id") and "__" not in str(c)
        and pd.api.types.is_numeric_dtype(X[c])
    ]
    for c in cand_pool:
        if c in seen_gc:
            continue
        cs = str(c)
        if "__json__" in cs or ("__" in cs and not cs.lower().endswith("_id")):
            continue
        nu = int(X[c].nunique(dropna=True))
        if 10 <= nu <= 50_000 and nu <= 0.85 * n_tr:
            g_cols.append((c, nu))
            seen_gc.add(c)
    if len(g_cols) < 2:
        return X, Xt
    # Rank true many-to-many entity pairs (prefer *_id columns, exclude 1:1 functional lookups)
    pairs = []
    str_cache = {}

    def _codes_tr(col):
        if col not in str_cache:
            s_tr = X[col].astype("string")
            codes, uniq = pd.factorize(s_tr, sort=False)
            str_cache[col] = (codes.astype(np.int64), uniq)
        return str_cache[col]

    for i in range(len(g_cols)):
        c1, nu1 = g_cols[i]
        id1 = str(c1).lower().endswith("_id")
        ca, _ = _codes_tr(c1)
        for j in range(i + 1, len(g_cols)):
            c2, nu2 = g_cols[j]
            id2 = str(c2).lower().endswith("_id")
            if not (id1 or id2):
                continue
            cb, _ = _codes_tr(c2)
            ok = (ca >= 0) & (cb >= 0)
            if int(ok.sum()) < 200:
                continue
            nb = nu2 + 1
            ekeys = ca[ok] * nb + cb[ok]
            n_edges = int(len(np.unique(ekeys)))
            max_n = max(nu1, nu2)
            # Require genuine dense many-to-many bipartite graph (not 1:1 functional attribute)
            if n_edges <= int(1.25 * max_n) or n_edges < 60:
                continue
            avg_deg = min(n_edges / max(nu1, 1), n_edges / max(nu2, 1))
            if avg_deg < 3.0:
                continue
            score_p = (2 if (id1 and id2) else 1, float(min(nu1, nu2)) * min(avg_deg, 20.0))
            pairs.append((score_p, c1, c2, nu1, nu2))
    if not pairs:
        return X, Xt
    pairs.sort(key=lambda x: x[0], reverse=True)

    tr_add, te_add = {}, {}
    svd_done = set()
    deg_done = set()
    rng = np.random.default_rng(42)
    half_id = rng.integers(0, 2, size=n_tr)

    def _fit_ppmi_svd(r_idx, c_idx, n1, n2, k, v0):
        C = sparse.coo_matrix((np.ones(len(r_idx), dtype=np.float64), (r_idx, c_idx)), shape=(n1, n2)).tocsr()
        C.sum_duplicates()
        total = float(C.data.sum())
        if total <= 0 or C.nnz < 20:
            return None
        deg1 = np.asarray(C.sum(axis=1)).ravel()
        deg2 = np.asarray(C.sum(axis=0)).ravel()
        udeg1 = np.diff(C.indptr).astype(np.float32)
        udeg2 = np.diff(C.tocsc().indptr).astype(np.float32)
        coo = C.tocoo()
        pmi = np.log((coo.data * total) / ((deg1[coo.row] + 1.0) * (deg2[coo.col] + 1.0)))
        pos = pmi > 0.0
        if int(pos.sum()) < max(15, k + 2):
            return None
        M = sparse.csr_matrix((pmi[pos].astype(np.float32), (coo.row[pos], coo.col[pos])), shape=(n1, n2))
        U, S, Vt = svds(M.astype(np.float64), k=k, v0=v0, maxiter=120, tol=1e-3)
        ord_s = np.argsort(S)[::-1]
        S = np.maximum(S[ord_s], 0.0)
        sq = np.sqrt(S)
        E1 = (U[:, ord_s] * sq[None, :]).astype(np.float32)
        E2 = (Vt[ord_s, :].T * sq[None, :]).astype(np.float32)
        E1n = E1 / np.maximum(np.linalg.norm(E1, axis=1, keepdims=True), 1e-6)
        E2n = E2 / np.maximum(np.linalg.norm(E2, axis=1, keepdims=True), 1e-6)
        return E1n, E2n, S, udeg1, udeg2, max(len(r_idx), 1)

    def _procrustes_align(E_sub: np.ndarray, E_ref: np.ndarray, active: np.ndarray) -> np.ndarray:
        if int(active.sum()) < E_ref.shape[1]:
            return E_sub
        Up, _, Vtp = np.linalg.svd((E_sub[active].T @ E_ref[active]).astype(np.float64))
        Q = (Up @ Vtp).astype(np.float32)
        return E_sub @ Q

    for _, c1, c2, nu1, nu2 in pairs[:2]:
        try:
            ca_tr, ua = _codes_tr(c1)
            cb_tr, ub = _codes_tr(c2)
            n1, n2 = len(ua), len(ub)
            k = min(6, min(n1, n2) - 2)
            if min(n1, n2) < 6 or k < 3:
                continue
            ok_tr = (ca_tr >= 0) & (cb_tr >= 0)
            v0 = rng.normal(size=min(n1, n2))
            full_fit = _fit_ppmi_svd(ca_tr[ok_tr], cb_tr[ok_tr], n1, n2, k, v0)
            if full_fit is None:
                continue
            E1n_full, E2n_full, S_full, udeg1_full, udeg2_full, n_fit_full = full_fit
            # Spectral-gap community gate: skip random unstructured bipartite graphs (flat Marchenko-Pastur spectrum)
            if (S_full[0] / max(float(S_full[-1]), 1e-6)) < 1.85:
                continue

            # 2-fold OOF scoring on train (eliminates self-edge memorization bias on cos and embeddings!)
            tr_cos = np.zeros(n_tr, dtype=np.float32)
            tr_deg1 = np.zeros(n_tr, dtype=np.float32)
            tr_deg2 = np.zeros(n_tr, dtype=np.float32)
            n_svd = min(3, k)
            tr_e1 = np.zeros((n_tr, n_svd), dtype=np.float32)
            tr_e2 = np.zeros((n_tr, n_svd), dtype=np.float32)
            oof_ok = True
            for fold_v in (0, 1):
                fit_m = ok_tr & (half_id != fold_v)
                val_m = half_id == fold_v
                sub_fit = _fit_ppmi_svd(ca_tr[fit_m], cb_tr[fit_m], n1, n2, k, v0)
                if sub_fit is None:
                    oof_ok = False
                    break
                E1n_s, E2n_s, _, ud1_s, ud2_s, n_fit_s = sub_fit
                E1n_al = _procrustes_align(E1n_s, E1n_full, ud1_s > 0)
                E2n_al = _procrustes_align(E2n_s, E2n_full, ud2_s > 0)
                va_idx = np.flatnonzero(val_m)
                ia, ib = ca_tr[va_idx], cb_tr[va_idx]
                ok_pair = (ia >= 0) & (ib >= 0)
                if ok_pair.any():
                    tr_cos[va_idx[ok_pair]] = np.sum(E1n_s[ia[ok_pair]] * E2n_s[ib[ok_pair]], axis=1)
                ok1, ok2 = ia >= 0, ib >= 0
                if ok1.any():
                    tr_deg1[va_idx[ok1]] = np.log1p(ud1_s[ia[ok1]] / float(n_fit_s) * 1000.0)
                    tr_e1[va_idx[ok1]] = E1n_al[ia[ok1], :n_svd]
                if ok2.any():
                    tr_deg2[va_idx[ok2]] = np.log1p(ud2_s[ib[ok2]] / float(n_fit_s) * 1000.0)
                    tr_e2[va_idx[ok2]] = E2n_al[ib[ok2], :n_svd]
            if not oof_ok:
                continue

            map_a = {v: i for i, v in enumerate(ua)}
            map_b = {v: i for i, v in enumerate(ub)}
            ca_te = Xt[c1].astype("string").map(map_a).fillna(-1).astype(np.int64).to_numpy()
            cb_te = Xt[c2].astype("string").map(map_b).fillna(-1).astype(np.int64).to_numpy()
            te_cos = np.zeros(n_te, dtype=np.float32)
            ok_te = (ca_te >= 0) & (cb_te >= 0)
            if ok_te.any():
                te_cos[ok_te] = np.sum(E1n_full[ca_te[ok_te]] * E2n_full[cb_te[ok_te]], axis=1)
            tr_add[f"gnn__{c1}__{c2}__cos"] = tr_cos
            te_add[f"gnn__{c1}__{c2}__cos"] = te_cos

            for col_nm, tr_deg_v, tr_e_v, ia_te, udeg_f, E_f in (
                (c1, tr_deg1, tr_e1, ca_te, udeg1_full, E1n_full),
                (c2, tr_deg2, tr_e2, cb_te, udeg2_full, E2n_full),
            ):
                if col_nm not in deg_done:
                    tr_add[f"gnn__{col_nm}__deg"] = tr_deg_v
                    te_add[f"gnn__{col_nm}__deg"] = np.where(
                        ia_te >= 0, np.log1p(udeg_f[np.maximum(ia_te, 0)] / float(n_fit_full) * 1000.0), 0.0
                    ).astype("float32")
                    deg_done.add(col_nm)
                if col_nm not in svd_done:
                    for d_i in range(n_svd):
                        tr_add[f"gnn__{col_nm}__svd{d_i}"] = tr_e_v[:, d_i]
                        te_add[f"gnn__{col_nm}__svd{d_i}"] = np.where(
                            ia_te >= 0, E_f[np.maximum(ia_te, 0), d_i], 0.0
                        ).astype("float32")
                    svd_done.add(col_nm)
        except Exception:
            log_exc(f"spectral_graph {c1}x{c2}")
            continue

    if tr_add:
        log(f"Zero-GNN OOF PPMI+SVD (строго train): +{len(tr_add)} графовых признаков")
        X = pd.concat([X, pd.DataFrame(tr_add, index=X.index)], axis=1)
        Xt = pd.concat([Xt, pd.DataFrame(te_add, index=Xt.index)], axis=1)
    return X, Xt


def prepare(train: pd.DataFrame, test: pd.DataFrame, spec) -> Prepared:
    if spec.task_type == "regression":
        y_num, _ = _parse_dirty_numeric(train[spec.target])
        y_all = y_num.to_numpy(dtype=float)
        valid_y_mask = np.isfinite(y_all)
        train = train.loc[valid_y_mask].reset_index(drop=True)
        y = y_all[valid_y_mask]
    else:
        valid_y_mask = train[spec.target].notna().to_numpy()
        train = train.loc[valid_y_mask].reset_index(drop=True)
        mapping = {c: i for i, c in enumerate(spec.classes)}
        y = train[spec.target].map(mapping).values
        if pd.isna(y).any():
            # значения таргета не из списка классов (например, 1 и 1.0): сопоставляем по строке
            smap = {str(c): i for i, c in enumerate(spec.classes)}
            y = train[spec.target].astype(str).map(smap).values
        y = np.asarray(y, dtype=int)
    test = test.reset_index(drop=True)
    exclude = {spec.target, spec.id_col}
    feats = [c for c in test.columns if c in train.columns and c not in exclude]

    X_parts, T_parts = [], []
    cat_cols, text_cols, raw_text, dropped = [], [], {}, []
    date_origins = {}
    json_unpacked = 0
    cat_hint = set(spec.cat_hint or [])
    for c in feats:
        a, b = train[c], test[c]
        name = str(c)
        try:
            if pd.api.types.is_bool_dtype(a) or pd.api.types.is_bool_dtype(b):
                X_parts.append(a.astype(float).rename(name).to_frame())
                T_parts.append(b.astype(float).rename(name).to_frame())
                continue
            if pd.api.types.is_datetime64_any_dtype(a):
                xa, xb = _expand_datetime(name, a, b)
                date_origins[f"{name}__ord"] = xa.attrs.get("origin")
                X_parts.append(xa)
                T_parts.append(xb)
                continue
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                ep_u = _unix_epoch_unit(name, a)
                if ep_u is not None:
                    xa, xb = _expand_datetime(
                        name,
                        pd.to_datetime(pd.to_numeric(a, errors="coerce"), unit=ep_u, errors="coerce", utc=True),
                        pd.to_datetime(pd.to_numeric(b, errors="coerce"), unit=ep_u, errors="coerce", utc=True),
                    )
                    date_origins[f"{name}__ord"] = xa.attrs.get("origin")
                    X_parts.append(xa)
                    T_parts.append(xb)
                    continue
                nu_a = a.nunique(dropna=True)
                is_bin01 = nu_a <= 2 and set(np.round(a.dropna().astype(float).unique(), 6)) <= {0.0, 1.0}
                if name in cat_hint and not is_bin01 and nu_a <= 1000 and _codes_seen(name, a, b):
                    cat_cols.append(name)

                    def fmt(s):
                        return s.astype(float).map(lambda v: None if pd.isna(v) else f"{v:.10g}").astype("string")

                    X_parts.append(fmt(a).rename(name).to_frame())
                    T_parts.append(fmt(b).rename(name).to_frame())
                else:
                    X_parts.append(a.astype("float32").rename(name).to_frame())
                    T_parts.append(pd.to_numeric(b, errors="coerce").astype("float32").rename(name).to_frame())
                continue
            sa, sb = a.astype("string"), b.astype("string")
            na, nn = _parse_dirty_numeric(sa)
            if nn > 0 and na.notna().sum() >= 0.95 * nn and (name not in cat_hint or not _codes_seen(name, sa, sb)):
                nb, _ = _parse_dirty_numeric(sb)
                X_parts.append(na.astype("float32").rename(name).to_frame())
                T_parts.append(nb.astype("float32").rename(name).to_frame())
                continue
            if _is_datetime_series(sa):
                xa, xb = _expand_datetime(name, sa, sb)
                date_origins[f"{name}__ord"] = xa.attrs.get("origin")
                X_parts.append(xa)
                T_parts.append(xb)
                continue
            if json_unpacked < 4:
                j_res = _unpack_flat_json(name, sa, sb)
                if j_res is not None:
                    ja, jb, j_cats = j_res
                    X_parts.append(ja)
                    T_parts.append(jb)
                    cat_cols.extend(j_cats)
                    json_unpacked += 1
                    continue
            lens = sa.dropna().str.len()
            words = sa.dropna().str.count(r"\s+") + 1
            is_text = len(lens) and words.mean() >= 3 and lens.mean() >= 15 and (
                sa.nunique() > 0.3 * len(sa) or (lens.mean() >= 40 and sa.nunique() > 0.05 * len(sa)))
            if is_text:
                text_cols.append(name)
                raw_text[name] = (sa.fillna(""), sb.fillna(""))
                for fn, f in (("len", lambda s: s.str.len()), ("words", lambda s: s.str.count(r"\s+") + 1),
                              ("digits", lambda s: s.str.count(r"\d")), ("upper", lambda s: s.str.count(r"[A-ZА-Я]"))):
                    X_parts.append(f(sa.fillna("")).astype("float32").rename(f"{name}__{fn}").to_frame())
                    T_parts.append(f(sb.fillna("")).astype("float32").rename(f"{name}__{fn}").to_frame())
                continue
            if sa.nunique() > 0.95 * max(1, sa.notna().sum()) and len(sa) > 200:
                # строковый идентификатор (почти уникален в каждой строке) — шум, а не признак
                dropped.append(name)
                continue
            cat_cols.append(name)
            X_parts.append(sa.rename(name).to_frame())
            T_parts.append(sb.rename(name).to_frame())
        except Exception:
            log_exc(f"prepare column {name}")
            dropped.append(name)

    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=train.index)
    Xt = pd.concat(T_parts, axis=1) if T_parts else pd.DataFrame(index=test.index)
    X.columns = [str(c) for c in X.columns]
    Xt.columns = [str(c) for c in Xt.columns]

    # Разности между парами дат с точным учётом origin каждой колонки (возраст аккаунта, дней с начала кампании)
    ord_cols = [c for c in X.columns if c.endswith("__ord")]
    if len(ord_cols) >= 2:
        main_ord = next((c for c in ord_cols if "__" not in c[:-5]), ord_cols[0])
        main_o = date_origins.get(main_ord)
        for oc in ord_cols:
            if oc == main_ord:
                continue
            dname = f"{main_ord[:-5]}__minus__{oc[:-5]}"
            if dname not in X.columns and len(X.columns) < 600:
                other_o = date_origins.get(oc)
                shift_days = ((main_o - other_o).total_seconds() / 86400.0) if (main_o is not None and other_o is not None) else 0.0
                X[dname] = ((X[main_ord].astype("float64") - X[oc].astype("float64")) + shift_days).astype("float32")
                Xt[dname] = ((Xt[main_ord].astype("float64") - Xt[oc].astype("float64")) + shift_days).astype("float32")

    # Кросс-табличный матчинг многозначных тег-колонок (запятые, ';', '|', Postgres-массивы '{a,b}'): число тегов, пересечение и Жаккар
    def _split_tag_tokens(val, delim: str):
        sv = str(val).strip()
        if len(sv) >= 2 and ((sv[0] == "{" and sv[-1] == "}") or (sv[0] == "[" and sv[-1] == "]")):
            sv = sv[1:-1]
        return [t.strip().strip("\"'").lower() for t in sv.split(delim) if t.strip().strip("\"'")]

    tag_vocab = {}
    tag_delim = {}
    for c in cat_cols:
        sample_s = X[c].dropna().astype(str).head(2000)
        if len(sample_s) < 20:
            continue
        is_pg_arr = (sample_s.str.strip().str.startswith("{") & sample_s.str.strip().str.endswith("}")).mean() >= 0.50
        best_d, best_rate = None, 0.0
        for d in (",", ";", "|"):
            rate = float(sample_s.str.contains(d, regex=False).mean())
            if rate > best_rate:
                best_rate, best_d = rate, d
        if best_d is None or (best_rate < 0.15 and not is_pg_arr):
            if is_pg_arr:
                best_d = ","
            else:
                continue
        tok_lists = [_split_tag_tokens(s, best_d) for s in sample_s]
        avg_len = float(np.mean([len(tl) for tl in tok_lists]))
        vocab = {t for tl in tok_lists for t in tl}
        if 1.1 <= avg_len <= 10.0 and 3 <= len(vocab) <= 200:
            tag_vocab[c] = vocab
            tag_delim[c] = best_d
            for df_t in (X, Xt):
                s = df_t[c].astype("string")
                stripped = s.str.strip().str.replace(r"^[\{\[]|[\}\]]$", "", regex=True).str.strip()
                cnt = stripped.str.count(re.escape(best_d)).astype("float32") + 1.0
                cnt = cnt.where(s.notna() & (stripped != ""), 0.0)
                df_t[f"{c}__n_tags"] = cnt.astype("float32")
    tag_cols = list(tag_vocab)
    for i, c1 in enumerate(tag_cols):
        for c2 in tag_cols[i + 1:]:
            v1, v2 = tag_vocab[c1], tag_vocab[c2]
            if len(v1 & v2) < max(2, int(0.3 * min(len(v1), len(v2)))):
                continue

            d1, d2 = tag_delim[c1], tag_delim[c2]

            def _tag_pair(sa: pd.Series, sb: pd.Series):
                ca, ua = pd.factorize(sa.astype("string"), sort=False)
                cb, ub = pd.factorize(sb.astype("string"), sort=False)
                sets_a = [frozenset(_split_tag_tokens(u, d1)) for u in ua]
                sets_b = [frozenset(_split_tag_tokens(u, d2)) for u in ub]
                n_row = len(sa)
                ov = np.full(n_row, np.nan, dtype=np.float32)
                jac = np.full(n_row, np.nan, dtype=np.float32)
                ok = (ca >= 0) & (cb >= 0)
                if not ok.any():
                    return ov, jac
                nb = len(ub) + 1
                pair_keys = ca[ok].astype(np.int64) * nb + cb[ok].astype(np.int64)
                uniq_p, inv = np.unique(pair_keys, return_inverse=True)
                ov_u = np.full(len(uniq_p), np.nan, dtype=np.float32)
                jac_u = np.full(len(uniq_p), np.nan, dtype=np.float32)
                for idx_p, pk in enumerate(uniq_p.tolist()):
                    s1 = sets_a[pk // nb]
                    s2 = sets_b[pk % nb]
                    if s1 and s2:
                        k = len(s1 & s2)
                        ov_u[idx_p] = float(k)
                        jac_u[idx_p] = float(k) / float(len(s1 | s2))
                ov[ok] = ov_u[inv]
                jac[ok] = jac_u[inv]
                return ov, jac

            ov_a, jac_a = _tag_pair(X[c1], X[c2])
            ov_b, jac_b = _tag_pair(Xt[c1], Xt[c2])
            X[f"{c1}__ov__{c2}"], Xt[f"{c1}__ov__{c2}"] = ov_a, ov_b
            X[f"{c1}__jac__{c2}"], Xt[f"{c1}__jac__{c2}"] = jac_a, jac_b

    # Кросс-табличный матчинг обычных категориальных колонок из разных таблиц (пересечение словарей train >= 30%)
    def _tbl_pfx(cname: str) -> str:
        s = str(cname)
        return s.split("__", 1)[0] if "__" in s else "__main__"

    plain_vocabs = {}
    for c in cat_cols:
        if c in tag_vocab:
            continue
        u_tr = X[c].dropna().unique()
        if not (2 <= len(u_tr) <= 500):
            continue
        v_tr = {str(v).strip().lower() for v in u_tr if str(v).strip()}
        if 2 <= len(v_tr) <= 500:
            plain_vocabs[c] = v_tr
    match_pairs = []
    plain_cols = list(plain_vocabs)
    for i, c1 in enumerate(plain_cols):
        p1, v1 = _tbl_pfx(c1), plain_vocabs[c1]
        for c2 in plain_cols[i + 1:]:
            if _tbl_pfx(c2) == p1:
                continue
            v2 = plain_vocabs[c2]
            inter = v1 & v2
            if len(inter) >= 2 and len(inter) / float(min(len(v1), len(v2))) >= 0.30:
                jac_v = len(inter) / float(len(v1 | v2))
                match_pairs.append((jac_v, len(inter), c1, c2))
    if match_pairs:
        match_pairs.sort(reverse=True)
        norm_cache = {}

        def _norm_cat_col(df_t: pd.DataFrame, col: str, tag: str):
            key = (tag, col)
            if key not in norm_cache:
                s = df_t[col]
                u = s.dropna().unique()
                mp = {v: str(v).strip().lower() for v in u}
                norm_cache[key] = s.map(mp)
            return norm_cache[key]

        eq_tr_dict, eq_te_dict = {}, {}
        for _, _, c1, c2 in match_pairs[:16]:
            a1, a2 = _norm_cat_col(X, c1, "tr"), _norm_cat_col(X, c2, "tr")
            b1, b2 = _norm_cat_col(Xt, c1, "te"), _norm_cat_col(Xt, c2, "te")
            ok_a = a1.notna().to_numpy() & a2.notna().to_numpy()
            ok_b = b1.notna().to_numpy() & b2.notna().to_numpy()
            eq_tr_dict[f"{c1}__eq__{c2}"] = np.where(ok_a, (a1 == a2).to_numpy(dtype=np.float32), np.nan).astype("float32")
            eq_te_dict[f"{c1}__eq__{c2}"] = np.where(ok_b, (b1 == b2).to_numpy(dtype=np.float32), np.nan).astype("float32")
        if eq_tr_dict:
            X = pd.concat([X, pd.DataFrame(eq_tr_dict, index=X.index)], axis=1)
            Xt = pd.concat([Xt, pd.DataFrame(eq_te_dict, index=Xt.index)], axis=1)

    # Zero-GNN OOF спектральные графовые эмбеддинги (строго на non-temporal сплитах: нулевая временная утечка/сдвиг)
    if detect_time_split(X, Xt, cat_cols) is None:
        X, Xt = _add_spectral_graph_features(X, Xt, cat_cols)

    # категориальные: частоты/ранги высокой кардинальности строго по train (без утечки частот из test)
    final_cats = []
    for c in cat_cols:
        sa = X[c].astype("string")
        sb = Xt[c].astype("string")
        vc_tr = sa.value_counts(dropna=True)
        if len(vc_tr) > 1000:
            rank = {v: i for i, v in enumerate(vc_tr.index)}
            tr_n = max(len(X), 1)
            X[c] = sa.map(rank).fillna(-1.0).astype("float32").values
            Xt[c] = sb.map(rank).fillna(-1.0).astype("float32").values
            X[c + "__freq"] = (sa.map(vc_tr).fillna(0.0) / tr_n).astype("float32").values
            Xt[c + "__freq"] = (sb.map(vc_tr).fillna(0.0) / tr_n).astype("float32").values
        else:
            seen = set(vc_tr.index)
            extra_te = [v for v in sb.dropna().unique() if v not in seen]
            cats = pd.CategoricalDtype(categories=list(vc_tr.index) + extra_te)
            X[c] = pd.Categorical(sa.astype(object), dtype=cats)
            Xt[c] = pd.Categorical(sb.astype(object), dtype=cats)
            final_cats.append(c)

    # константы и полностью пустые колонки
    for c in list(X.columns):
        if X[c].nunique(dropna=False) <= 1:
            dropped.append(c)
    X = X.drop(columns=[c for c in dropped if c in X.columns])
    Xt = Xt.drop(columns=[c for c in dropped if c in Xt.columns])
    final_cats = [c for c in final_cats if c in X.columns]

    # безопасные имена: LightGBM/XGBoost падают на [ ] { } : , " < > в именах признаков
    rename, used = {}, set()
    for c in X.columns:
        n = re.sub(r"[^\w]+", "_", str(c)).strip("_") or "f"
        while n in used:
            n += "_"
        used.add(n)
        rename[c] = n
    X = X.rename(columns=rename)
    Xt = Xt.rename(columns=rename)
    final_cats = [rename.get(c, c) for c in final_cats]
    text_cols = [rename.get(c, c) for c in text_cols]
    raw_text = {rename.get(k, k): v for k, v in raw_text.items()}

    prep = Prepared(X=X, Xt=Xt, y=y, cat_cols=final_cats, text_cols=text_cols, raw_text=raw_text)
    prep.info = {"n_train": len(X), "n_test": len(Xt), "n_features": X.shape[1], "cat_cols": final_cats,
                 "text_cols": text_cols, "dropped": dropped, "valid_y_mask": valid_y_mask,
                 "rename": {k: v for k, v in rename.items() if k != v}}
    return prep


# ---------------------------------------------------------------- временное разбиение


def detect_time_split(X: pd.DataFrame, Xt: pd.DataFrame, cat_cols):
    """Колонка, по которой тест целиком лежит «после» обучения (временной сплит)."""
    best, best_score = None, 0.0
    columns = sorted(X.columns, key=lambda c: (not str(c).endswith("__ord"), str(c)))
    for c in columns:
        if c in cat_cols or not pd.api.types.is_numeric_dtype(X[c]):
            continue
        a, b = X[c].dropna(), Xt[c].dropna()
        if len(b) < 0.9 * len(Xt) or len(a) < 0.9 * len(X) or a.nunique() < 2:
            continue
        above = float((b > a.max()).mean())
        if above >= 0.95 and above > best_score:
            best, best_score = c, above
    return best


def time_periods(values: pd.Series, max_periods: int = 60):
    v = values.astype(float)
    if v.nunique() <= max_periods:
        return v
    q = np.unique(np.nanquantile(v, np.linspace(0, 1, 11)))
    return pd.Series(np.searchsorted(q[1:-1], v.values, side="right"), index=v.index).astype(float)


# ---------------------------------------------------------------- признаки-преобразования


def op_freq(X, Xt, cols, train_only: bool = False):
    out_a, out_b = {}, {}
    n_tr = max(len(X), 1)
    for c in cols:
        if train_only:
            sa, sb = X[c].astype("string"), Xt[c].astype("string")
            vc = sa.value_counts(dropna=False) / n_tr
            out_a[f"{c}__cnt"] = sa.map(vc).fillna(0.0).astype("float32").values
            out_b[f"{c}__cnt"] = sb.map(vc).fillna(0.0).astype("float32").values
        else:
            both = pd.concat([X[c], Xt[c]], ignore_index=True)
            vc = both.astype("string").value_counts(dropna=False)
            f = both.astype("string").map(vc).astype("float32").values
            out_a[f"{c}__cnt"] = f[: len(X)]
            out_b[f"{c}__cnt"] = f[len(X):]
    return pd.DataFrame(out_a), pd.DataFrame(out_b)



def op_nan_count(X, Xt):
    return (pd.DataFrame({"row__nan": X.isna().sum(1).astype("float32").values}),
            pd.DataFrame({"row__nan": Xt.isna().sum(1).astype("float32").values}))


def op_dup_count(X, Xt, train_only: bool = False):
    if train_only:
        ha = pd.util.hash_pandas_object(X.astype("string"), index=False)
        hb = pd.util.hash_pandas_object(Xt.astype("string"), index=False)
        vc = ha.value_counts()
        return (pd.DataFrame({"row__dup": ha.map(vc).fillna(0.0).astype("float32").values}),
                pd.DataFrame({"row__dup": hb.map(vc).fillna(0.0).astype("float32").values}))
    both = pd.concat([X, Xt], ignore_index=True)
    h = pd.util.hash_pandas_object(both.astype("string"), index=False)
    cnt = h.map(h.value_counts()).astype("float32").values
    return pd.DataFrame({"row__dup": cnt[: len(X)]}), pd.DataFrame({"row__dup": cnt[len(X):]})


def op_group_dev(X, Xt, pairs, train_only: bool = False):
    out_a, out_b = {}, {}
    for cat, num in pairs:
        if train_only:
            ca, cb = X[cat].astype("string"), Xt[cat].astype("string")
            g = X.groupby(ca, dropna=False)[num].agg(["mean", "std"])
            ma, sa = ca.map(g["mean"]), ca.map(g["std"])
            mb, sb = cb.map(g["mean"]), cb.map(g["std"])
            out_a[f"{num}__z_by_{cat}"] = ((X[num] - ma) / (sa + 1e-9)).astype("float32").values
            out_b[f"{num}__z_by_{cat}"] = ((Xt[num] - mb) / (sb + 1e-9)).astype("float32").values
            out_a[f"{num}__r_by_{cat}"] = (X[num] / (ma.abs() + 1e-9)).astype("float32").values
            out_b[f"{num}__r_by_{cat}"] = (Xt[num] / (mb.abs() + 1e-9)).astype("float32").values
        else:
            both = pd.concat([X[[cat, num]], Xt[[cat, num]]], ignore_index=True)
            g = both.groupby(both[cat].astype("string"), dropna=False)[num]
            mean = g.transform("mean")
            std = g.transform("std")
            dev = ((both[num] - mean) / (std + 1e-9)).astype("float32").values
            rat = (both[num] / (mean.abs() + 1e-9)).astype("float32").values
            out_a[f"{num}__z_by_{cat}"], out_b[f"{num}__z_by_{cat}"] = dev[: len(X)], dev[len(X):]
            out_a[f"{num}__r_by_{cat}"], out_b[f"{num}__r_by_{cat}"] = rat[: len(X)], rat[len(X):]
    return pd.DataFrame(out_a), pd.DataFrame(out_b)



def _key_hash(df: pd.DataFrame, cols):
    return pd.util.hash_pandas_object(df[cols].astype("string"), index=False).values


def entity_periods(X, Xt, time_col):
    """Номера периодов: train — по значениям/квантилям времени, весь test — следующий период после train.

    Строки одного периода не видят таргеты друг друга, как и строки теста — поэтому признаки истории
    на валидации по времени устроены так же, как на тесте (без утечки внутри периода).
    """
    per = time_periods(X[time_col]).astype(float)
    return per.values, np.full(len(Xt), np.nanmax(per.values) + 1.0)


def op_entity_history(X, Xt, y, key_cols, time_col, t_train=None, t_test=None):
    """История таргета того же объекта в прошлых периодах (без заглядывания в будущее).

    Для строки периода t берутся только наблюдения обучающей выборки с временем < t: значение в последнем
    периоде, среднее, число наблюдений, разрыв во времени и тренд. Безопасно и для валидации по времени.
    """
    n = len(X)
    both = pd.concat([X[key_cols], Xt[key_cols]], ignore_index=True)
    key = _key_hash(both, key_cols)
    if t_train is None:
        t_train, t_test = X[time_col].astype(float).values, Xt[time_col].astype(float).values
    t = np.concatenate([np.asarray(t_train, float), np.asarray(t_test, float)])
    yy = np.concatenate([np.asarray(y, dtype=float), np.full(len(Xt), np.nan)])
    hist = pd.DataFrame({"key": key[:n], "t": t[:n], "y": yy[:n]}).dropna()
    agg = hist.groupby(["key", "t"], sort=True).y.agg(["mean", "size"]).reset_index()
    agg["csum"] = (agg["mean"] * agg["size"]).groupby(agg["key"]).cumsum()
    agg["ccnt"] = agg["size"].groupby(agg["key"]).cumsum()
    agg["first"] = agg.groupby("key")["mean"].transform("first")
    agg["first_t"] = agg.groupby("key")["t"].transform("first")
    agg = agg.rename(columns={"t": "t_last", "mean": "last"})
    agg["t_join"] = agg["t_last"]
    left = pd.DataFrame({"key": key, "t_join": t, "row": np.arange(len(t))}).dropna(subset=["t_join"])
    left = left.sort_values("t_join")
    agg = agg.sort_values("t_join")
    m = pd.merge_asof(left, agg[["key", "t_join", "t_last", "last", "csum", "ccnt", "first", "first_t"]],
                      on="t_join", by="key", allow_exact_matches=False)
    m = m.sort_values("row")
    res = pd.DataFrame(index=np.arange(len(t)))
    res.loc[m.row.values, "hist__last"] = m["last"].values
    res.loc[m.row.values, "hist__mean"] = (m["csum"] / m["ccnt"]).values
    res.loc[m.row.values, "hist__cnt"] = m["ccnt"].values
    res.loc[m.row.values, "hist__gap"] = (m["t_join"] - m["t_last"]).values
    span = (m["t_last"] - m["first_t"]).replace(0, np.nan)
    res.loc[m.row.values, "hist__trend"] = ((m["last"] - m["first"]) / span).values
    # Строго каузальное байесовски сглаженное среднее (m=20 к прошлому глобальному среднему по периодам < t)
    g_per = hist.groupby("t", sort=True)["y"].agg(["sum", "size"]).reset_index().rename(columns={"t": "t_join"})
    g_per["gmean"] = g_per["sum"].cumsum() / g_per["size"].cumsum()
    mg = pd.merge_asof(left[["t_join", "row"]], g_per[["t_join", "gmean"]].sort_values("t_join"),
                       on="t_join", allow_exact_matches=False).sort_values("row")
    gmean = np.full(len(t), np.nan, dtype=np.float64)
    gmean[mg.row.values] = mg["gmean"].values
    csum_arr = np.zeros(len(t), dtype=np.float64)
    ccnt_arr = np.zeros(len(t), dtype=np.float64)
    csum_arr[m.row.values] = np.nan_to_num(m["csum"].to_numpy(dtype=np.float64), nan=0.0)
    ccnt_arr[m.row.values] = np.nan_to_num(m["ccnt"].to_numpy(dtype=np.float64), nan=0.0)
    prior_m = 20.0
    res["hist__smooth"] = np.where(np.isfinite(gmean), (csum_arr + prior_m * gmean) / (ccnt_arr + prior_m), np.nan)
    res = res.astype("float32")
    return res.iloc[:n].reset_index(drop=True), res.iloc[n:].reset_index(drop=True)


_DERIVED_SFX = ("__year", "__month", "__day", "__dow", "__doy", "__hour", "__minute", "__ord", "__cnt", "__freq",
                "__len", "__words", "__digits", "__upper", "__matched", "__raw_nulls", "__lag", "__dlag",
                "__d1", "__rm3", "__h7d_tgt", "__dow4_tgt", "__anom", "__fcst", "__runlen",
                "__sys_mean", "__sys_max", "__sys_dev", "__n_tags",
                "__hist_nunique_1d", "__hist_nunique_7d", "__hist_nunique_30d",
                "__hist_cnt_1h", "__hist_cnt_6h", "__hist_burst_1h_1d")
_DERIVED_PFX = ("stream__", "stream_grp__", "nc__", "nc_d1__", "nc_prof__", "nc_gb__", "nc_gbd__", "nc_interp__", "gnn__")


def entity_lag_cols(X, y, key_cols, t_train, exclude=(), max_cols: int = 4, min_corr: float = 0.15):
    """Числовые признаки, изменение которых у объекта между его периодами связано с изменением таргета
    (|corr(Δx, Δy)| по соседним наблюдениям одного объекта в train). Берутся только сильные: не слабее
    min_corr и половины лучшего. На WiDS: рейтинг ENERGY STAR (0.42) — да, погода (~0.1, 50 похожих колонок) —
    нет: с ней RMSE 34.90 против 34.67 без неё."""
    skip = set(key_cols) | set(exclude)
    cols = [c for c in X.columns if c not in skip and not str(c).endswith(_DERIVED_SFX)
            and not str(c).startswith(_DERIVED_PFX)
            and "__minus__" not in str(c) and "__ov__" not in str(c) and "__jac__" not in str(c)
            and "__eq__" not in str(c)
            and pd.api.types.is_numeric_dtype(X[c])
            and X[c].nunique() > 2]
    if not cols:
        return []
    key = _key_hash(X, key_cols)
    rows = np.arange(len(X))
    if len(X) > 300_000:
        # на больших данных — часть объектов целиком (их ряды по периодам не рвутся)
        rows = rows[key % max(2, len(X) // 300_000) == 0]
    df = X[cols].iloc[rows].astype("float32").reset_index(drop=True)
    df["__key"] = key[rows]
    df["__t"] = np.asarray(t_train, float)[rows]
    df["__y"] = np.asarray(y, float)[rows]
    g = df.dropna(subset=["__t", "__y"]).groupby(["__key", "__t"], sort=True).mean()
    if len(g) < 200:
        return []
    keys = g.index.get_level_values(0).values
    same = np.r_[False, keys[1:] == keys[:-1]]  # соседняя строка — прошлый период того же объекта
    d = g.diff()[same]
    if len(d) < 100:
        return []
    out = []
    for c in cols:
        dc, dy = d[c].values, d["__y"].values
        ok = np.isfinite(dc) & np.isfinite(dy)
        if ok.sum() < 100 or (np.abs(dc[ok]) > 1e-9).mean() < 0.05 or np.std(dc[ok]) == 0:
            continue
        r = abs(float(np.corrcoef(dc[ok], dy[ok])[0, 1]))
        if np.isfinite(r) and r >= min_corr:
            out.append((r, c))
    out = sorted(out, reverse=True)
    return [c for r, c in out[:max_cols] if r >= 0.5 * out[0][0]]


def op_entity_lags(X, Xt, key_cols, cols, t_train, t_test):
    """Значение признака у того же объекта в его прошлом периоде и изменение с тех пор: изменение рейтинга или
    нагрузки объекта объясняет изменение таргета, которого история одного таргета не видит."""
    n = len(X)
    both = pd.concat([X[key_cols], Xt[key_cols]], ignore_index=True)
    key = _key_hash(both, key_cols)
    t = np.concatenate([np.asarray(t_train, float), np.asarray(t_test, float)])
    left = pd.DataFrame({"key": key, "t_join": t, "row": np.arange(len(t))}).dropna(subset=["t_join"])
    left = left.sort_values("t_join")
    res = pd.DataFrame(index=np.arange(len(t)))
    for c in cols:
        v = np.concatenate([X[c].astype(float).values, Xt[c].astype(float).values])
        hist = pd.DataFrame({"key": key[:n], "t_join": t[:n], "v": v[:n]}).dropna()
        agg = hist.groupby(["key", "t_join"], sort=True).v.mean().reset_index().sort_values("t_join")
        m = pd.merge_asof(left, agg, on="t_join", by="key", allow_exact_matches=False).sort_values("row")
        res.loc[m.row.values, f"{c}__lag"] = m["v"].values
        res[f"{c}__dlag"] = v - res[f"{c}__lag"].values
    res = res.astype("float32")
    return res.iloc[:n].reset_index(drop=True), res.iloc[n:].reset_index(drop=True)


def entity_key_candidates(X: pd.DataFrame, time_col: str, cat_cols, extra=None, max_keys: int = 4):
    """Кандидаты в ключ объекта: колонки, значения которых «переживают» смену периода."""
    tper = pd.Series(time_periods(X[time_col]).values, index=X.index)
    periods = np.sort(tper.dropna().unique())
    if len(periods) < 2:
        return []
    pairs = list(zip(periods[:-1], periods[1:]))[-6:]
    stats = {}
    for c in X.columns:
        cs = str(c)
        if (c == time_col or cs.endswith(_DERIVED_SFX) or cs.startswith(_DERIVED_PFX)
                or "__minus__" in cs or "__ov__" in cs or "__jac__" in cs or "__eq__" in cs):
            continue
        # Joined continuous numeric telemetry columns are never entity keys
        if "__" in cs and pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 30:
            continue
        s = X[c]
        pers = []
        for p0, p1 in pairs:
            a = set(s[tper == p0].dropna().astype(str).unique())
            b = s[tper == p1].dropna().astype(str)
            if len(b) and a:
                pers.append(b.isin(a).mean())
        if pers:
            stats[c] = (float(np.mean(pers)), int(s.nunique()))

    def quality(cols):
        if not cols:
            return 0.0, 0.0
        k = _key_hash(X, cols)
        df = pd.DataFrame({"k": k, "t": tper.values})
        uniq = df.groupby("t").k.nunique().sum() / max(1, len(df))
        match = []
        for p0, p1 in pairs:
            a = set(df.k[df.t == p0])
            b = df.k[df.t == p1]
            if len(b):
                match.append(b.isin(a).mean())
        return float(uniq), float(np.mean(match)) if match else 0.0

    # Предпочитаем исходные колонки главной таблицы (без '__' из джойненных сенсоров/справочников)
    static_all = [c for c, (p, nu) in stats.items() if p >= 0.6]
    static_main = [c for c in static_all if "__" not in str(c)]
    static = static_main if static_main else static_all
    cands = []
    if extra:
        ex = [c for c in extra if c in X.columns and c != time_col][:3]
        if ex:
            cands.append(ex)
    # жадно (до 3 колонок): добавляем колонки, пока заметно растёт uniq*match
    order = sorted(static, key=lambda c: -stats[c][1])
    cur, cur_q = [], 0.0
    for c in order:
        if len(cur) >= 3:
            break
        u, m = quality(cur + [c])
        q = u * m
        if q > cur_q + (1e-4 if not cur else 0.01):
            cur, cur_q = cur + [c], q
    if cur:
        cands.append(cur)
        # тот же ключ, уточнённый одной стабильной категорией (итого <= 4 колонок)
        cats_static = sorted([c for c in static if c in set(cat_cols) and c not in cur and stats[c][0] >= 0.9],
                             key=lambda c: -stats[c][1])
        if cats_static and len(cur) <= 3:
            cands.append(cur + cats_static[:1])
    if 2 <= len(static) <= 4:
        cands.append(static)
    out = []
    for c in cands:
        u, m = quality(c)
        if m >= 0.05 and u >= 0.3 and c not in out:
            out.append(c)
            log(f"ключ объекта {c}: уникальность в периоде {u:.3f}, совпадение с прошлым периодом {m:.3f}")
    out = out[:max_keys]
    # объект встречается много раз за период (например, магазин по дням): история по одной колонке-группе
    # = кодирование таргета только по прошлым периодам
    per_rows = len(X) / max(1, len(periods))
    groups = [c for c in static if stats[c][0] >= 0.8 and 10 <= stats[c][1] <= 0.5 * per_rows]
    for c in sorted(groups, key=lambda c: -stats[c][1])[:2]:
        if [c] not in out:
            out.append([c])
            log(f"группа для истории таргета {c}: {stats[c][1]} значений, сохраняемость {stats[c][0]:.2f}")
    return out


def op_text_oof(raw_text: dict, y, task_type: str, n_classes: int, n_jobs: int = 1, seed: int = 0, periods=None):
    """TF-IDF + линейная модель по каждому тексту: OOF-прогнозы для train и прогноз для test как признаки."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold

    out_a, out_b = {}, {}
    y = np.asarray(y)
    for name, (sa, sb) in raw_text.items():
        sa_s = sa.astype(str)
        sb_s = sb.astype(str)
        if periods is not None:
            # Strictly causal temporal TF-IDF: fit fresh word/char vectorizers INSIDE each fold on periods < p
            # (zero future-period AND zero test vocabulary/IDF leakage), and on train-only for test.
            per = np.asarray(periods, dtype=float)
            if len(per) != len(y):
                raise ValueError("Text periods must align with target rows")
            total_chars = float(sa_s.str.len().sum())
            big = len(sa_s) > 150_000 or total_chars > 1.5e8
            if task_type == "regression":
                mk = lambda: Ridge(alpha=3.0)  # noqa: E731
                k_out = 1
            else:
                mk = lambda: LogisticRegression(C=4.0, max_iter=300 if len(sa_s) < 150_000 else 100, n_jobs=n_jobs)  # noqa: E731
                k_out = 1 if n_classes == 2 else n_classes
            folds = [(np.flatnonzero(per < value), np.flatnonzero(per == value))
                     for value in np.sort(np.unique(per[np.isfinite(per)]))]

            def _fit_transform_pair(tr_texts, ev_texts):
                w = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if not big else 5,
                                    max_features=200_000 if not big else 100_000, sublinear_tf=True, dtype=np.float32)
                tr_m = [w.fit_transform(tr_texts)]
                ev_m = [w.transform(ev_texts)]
                if not big:
                    ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000,
                                         sublinear_tf=True, dtype=np.float32)
                    tr_m.append(ch.fit_transform(tr_texts))
                    ev_m.append(ch.transform(ev_texts))
                return sparse.hstack(tr_m).tocsr(), sparse.hstack(ev_m).tocsr()

            def _pred_causal(tr_idx, ev_texts):
                n_ev = len(ev_texts)
                if not len(tr_idx):
                    return np.full((n_ev, k_out), 0.0 if task_type == "regression" else 1 / n_classes)
                if task_type != "regression" and len(np.unique(y[tr_idx])) < 2:
                    p = np.zeros((n_ev, n_classes))
                    p[:, int(y[tr_idx][0])] = 1.0
                    return p[:, 1:] if n_classes == 2 else p
                try:
                    Atr, Aev = _fit_transform_pair(sa_s.iloc[tr_idx], ev_texts)
                except ValueError:
                    return np.full((n_ev, k_out), 0.0 if task_type == "regression" else 1 / n_classes)
                m = mk().fit(Atr, y[tr_idx])
                if task_type == "regression":
                    return m.predict(Aev).reshape(-1, 1)
                p = np.zeros((n_ev, n_classes))
                p[:, m.classes_.astype(int)] = m.predict_proba(Aev)
                return p[:, 1:] if n_classes == 2 else p

            oof = np.full((len(sa_s), k_out), 0.0 if task_type == "regression" else 1 / n_classes)
            for tr, va in folds:
                oof[va] = _pred_causal(tr, sa_s.iloc[va])
            tp = _pred_causal(np.arange(len(y)), sb_s)
            for j in range(k_out):
                out_a[f"{name}__tfidf{j}"] = oof[:, j].astype("float32")
                out_b[f"{name}__tfidf{j}"] = tp[:, j].astype("float32")
            continue

        both = pd.concat([sa_s, sb_s], ignore_index=True)
        total_chars = float(both.str.len().sum())
        big = len(both) > 150_000 or total_chars > 1.5e8
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if not big else 5,
                               max_features=200_000 if not big else 100_000, sublinear_tf=True, dtype=np.float32)
        mats = [word.fit_transform(both)]
        if not big:
            # символьные n-граммы ловят опечатки и морфологию, но дороги на больших корпусах
            char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000,
                                   sublinear_tf=True, dtype=np.float32)
            mats.append(char.fit_transform(both))
        M = sparse.hstack(mats).tocsr()
        A, B = M[: len(sa)], M[len(sa):]
        if task_type == "regression":
            mk = lambda: Ridge(alpha=3.0)  # noqa: E731
            folds = KFold(5, shuffle=True, random_state=seed).split(A)
            k_out = 1
        else:
            mk = lambda: LogisticRegression(C=4.0, max_iter=300 if len(sa) < 150_000 else 100, n_jobs=n_jobs)  # noqa: E731
            folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(A, y)
            k_out = 1 if n_classes == 2 else n_classes

        def predict_from(tr, target):
            if not len(tr):
                return np.full((target.shape[0], k_out), 0.0 if task_type == "regression" else 1 / n_classes)
            if task_type != "regression" and len(np.unique(y[tr])) < 2:
                p = np.zeros((target.shape[0], n_classes))
                p[:, int(y[tr][0])] = 1.0
                return p[:, 1:] if n_classes == 2 else p
            m = mk().fit(A[tr], y[tr])
            if task_type == "regression":
                return m.predict(target).reshape(-1, 1)
            p = np.zeros((target.shape[0], n_classes))
            p[:, m.classes_.astype(int)] = m.predict_proba(target)
            return p[:, 1:] if n_classes == 2 else p

        oof = np.full((A.shape[0], k_out), 0.0 if task_type == "regression" else 1 / n_classes)
        for tr, va in folds:
            oof[va] = predict_from(tr, A[va])
        tp = predict_from(np.arange(len(y)), B)
        for j in range(k_out):
            out_a[f"{name}__tfidf{j}"] = oof[:, j].astype("float32")
            out_b[f"{name}__tfidf{j}"] = tp[:, j].astype("float32")
    return pd.DataFrame(out_a), pd.DataFrame(out_b)



def op_pairwise(X, Xt, cols):
    """Отношения и разности пар важных числовых признаков (часто несут смысл: доли, нормировки)."""
    out_a, out_b = {}, {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            for df, out in ((X, out_a), (Xt, out_b)):
                va, vb = df[a].astype("float64"), df[b].astype("float64")
                out[f"{a}__div__{b}"] = (va / (vb.abs() + 1e-9) * np.sign(vb).replace(0, 1)).astype("float32").values
                out[f"{a}__sub__{b}"] = (va - vb).astype("float32").values
    A, B = pd.DataFrame(out_a), pd.DataFrame(out_b)
    A = A.replace([np.inf, -np.inf], np.nan)
    B = B.replace([np.inf, -np.inf], np.nan)
    return A, B


def op_target_oof(X, Xt, y, cols, folds, n_classes: int = 1, smooth: float = 20.0):
    """Кодирование категории средним таргетом (для классов — долями классов) только по чужим фолдам.

    Строка train получает статистику по фолдам, где её нет, тест — по всему train; сглаживание к общему
    среднему (smooth псевдострок). Полезно для категорий с сотнями значений, где разбиения деревьев слабее.
    """
    yy = np.asarray(y, dtype=float)
    targets = [yy] if n_classes <= 2 else [(yy == k).astype(float) for k in range(n_classes)]
    names = ["te"] if n_classes <= 2 else [f"te{k}" for k in range(n_classes)]
    out_a, out_b = {}, {}
    for c in cols:
        ka = X[c].astype("string").fillna("__nan__").values
        kb = Xt[c].astype("string").fillna("__nan__").values
        for t, nm in zip(targets, names):
            prior = float(np.mean(t))
            oof = np.full(len(X), prior, dtype=float)
            for tr, va in folds:
                g = pd.DataFrame({"k": ka[tr], "t": t[tr]}).groupby("k").t.agg(["sum", "count"])
                enc = (g["sum"] + smooth * prior) / (g["count"] + smooth)
                oof[va] = pd.Series(ka[va]).map(enc).fillna(prior).values
            g = pd.DataFrame({"k": ka, "t": t}).groupby("k").t.agg(["sum", "count"])
            enc = (g["sum"] + smooth * prior) / (g["count"] + smooth)
            out_a[f"{c}__{nm}"] = oof.astype("float32")
            out_b[f"{c}__{nm}"] = pd.Series(kb).map(enc).fillna(prior).values.astype("float32")
    return pd.DataFrame(out_a), pd.DataFrame(out_b)

def cross_impute_outages(X: pd.DataFrame, Xt: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generic cross-table sensor outage reconstruction using concurrent non-outage features.
    Triggered only when temporal joins emitted *__raw_nulls markers. Fit strictly on train non-outage rows (no y).
    """
    null_markers = [c for c in X.columns if str(c).endswith("__raw_nulls")]
    if not null_markers or len(X) < 300 or len(X) > 100_000:
        return X, Xt
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        return X, Xt
    X_out, Xt_out = X.copy(), Xt.copy()
    cat_set = set(cat_cols)
    num_all = [c for c in X_out.columns if c not in cat_set and pd.api.types.is_numeric_dtype(X_out[c])]
    cat_all = [c for c in cat_cols if c in X_out.columns]
    X_str, Xt_str = X_out.copy(), Xt_out.copy()
    for c in cat_all:
        X_str[c] = X_str[c].astype("string").fillna("__missing__").astype(str)
        Xt_str[c] = Xt_str[c].astype("string").fillna("__missing__").astype(str)
    reconstructed = 0
    _lag_sfx = ("__d1", "__rm3", "__h7d_tgt", "__dow4_tgt", "__anom", "__fcst", "__runlen",
                "__sys_mean", "__sys_max", "__sys_dev", "__ord", "__year")
    for nm in null_markers:
        prefix = nm[:-len("__raw_nulls")] + "__"
        t_cols = [c for c in num_all if c.startswith(prefix) and c != nm and c != prefix + "matched"
                  and not c.endswith(_lag_sfx)]
        other_num = [c for c in num_all if not c.startswith(prefix) and not c.startswith(_DERIVED_PFX)
                     and not c.endswith(_lag_sfx)]
        other_cat = [c for c in cat_all if not c.startswith(prefix)]
        if not t_cols or (not other_num and not other_cat):
            continue
        ok_tr = X_out[nm].fillna(0).values == 0
        bad_tr = ~ok_tr
        bad_te = Xt_out[nm].fillna(0).values > 0
        if ok_tr.sum() < 200 or (not bad_tr.any() and not bad_te.any()):
            continue
        try:
            tf = []
            if other_num:
                tf.append(("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other_num))
            if other_cat:
                tf.append(("c", OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=40, sparse_output=False), other_cat))
            pre = ColumnTransformer(tf)
            Z_ok = pre.fit_transform(X_str.loc[ok_tr])
            Z_btr = pre.transform(X_str.loc[bad_tr]) if bad_tr.any() else None
            Z_bte = pre.transform(Xt_str.loc[bad_te]) if bad_te.any() else None
            for tc in t_cols:
                y_ok = pd.to_numeric(X_out.loc[ok_tr, tc], errors="coerce").values
                fin = np.isfinite(y_ok)
                if fin.sum() < 200 or np.nanstd(y_ok[fin]) < 1e-8:
                    continue
                reg = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Z_ok[fin], y_ok[fin])
                lo, hi = float(np.nanmin(y_ok[fin])), float(np.nanmax(y_ok[fin]))
                if bad_tr.any():
                    p = np.clip(reg.predict(Z_btr), lo, hi)
                    cur = pd.to_numeric(X_out.loc[bad_tr, tc], errors="coerce").values
                    X_out.loc[bad_tr, tc] = np.where(np.isfinite(cur), 0.7 * p + 0.3 * cur, p).astype("float32")
                if bad_te.any():
                    p = np.clip(reg.predict(Z_bte), lo, hi)
                    cur = pd.to_numeric(Xt_out.loc[bad_te, tc], errors="coerce").values
                    Xt_out.loc[bad_te, tc] = np.where(np.isfinite(cur), 0.7 * p + 0.3 * cur, p).astype("float32")
                reconstructed += 1
        except Exception:
            log_exc(f"cross_impute {nm}")
    if reconstructed:
        log(f"кросс-сенсорная реконструкция сбоев: обновлено {reconstructed} колонок")
    return X_out, Xt_out


def causal_stream_features(train: pd.DataFrame, test: pd.DataFrame, exclude=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generic strictly-causal event-stream features (< t, NEVER >= t, NEVER using target or joined tables).

    Captures real-time arrival bursts, inter-arrival gaps, and recent-window event composition strictly before
    the current event timestamp `t` (identical in online single-event streaming). Protected by a chronological
    sampling-density guard so differing train/test subsampling rates never cause covariate shift.
    """
    import warnings

    excl = set(exclude or ()) | (set(train.columns) - set(test.columns))
    n_tr, n_te = len(train), len(test)
    n_tot = n_tr + n_te
    if n_tot < 500 or n_tot > 1_000_000:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    # Inspect ONLY raw main-table columns (no '__' from joined auxiliary tables).
    raw_cols = [c for c in test.columns if c in train.columns and c not in excl and "__" not in str(c)]
    time_cands = []
    for c in raw_cols:
        low = str(c).lower()
        if low.startswith(("start_", "end_")):
            continue
        is_dt = pd.api.types.is_datetime64_any_dtype(train[c])
        if not is_dt:
            if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c]):
                continue
            name_ok = (low in {"timestamp", "datetime", "event_time", "created_at", "recorded_at", "ts", "time"}
                       or low.endswith(("_time", "_timestamp", "_datetime", "_at", "_ts")))
            if not name_ok and not _is_datetime_series(train[c]):
                continue
        sample = train[c].dropna().head(300)
        if len(sample) < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = pd.to_datetime(sample, errors="coerce", utc=True)
        if p.notna().mean() >= 0.95 and (p.dt.hour != 0).mean() > 0.05:
            time_cands.append(c)
    if len(time_cands) != 1:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    tc = time_cands[0]
    comb = pd.concat([train[raw_cols], test[raw_cols]], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = pd.to_datetime(comb[tc], errors="coerce", utc=True)
    valid = ts.notna().values
    if float(valid.mean()) < 0.95:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    ts_ok = ts.loc[valid]
    if ts_ok.nunique() / max(len(ts_ok), 1) < 0.05:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    # Require sub-daily median positive inter-event gap (< 12h)
    ts_sorted_ns = np.sort(ts_ok.values.astype("int64"))
    diffs = np.diff(ts_sorted_ns)
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) < 100 or float(np.median(pos_diffs)) >= 12 * 3600 * 1e9:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    # Chronological density guard: skip if test is chronologically after train with mismatched sampling rate.
    tr_t = ts.iloc[:n_tr].dropna()
    te_t = ts.iloc[n_tr:].dropna()
    if len(tr_t) > 10 and len(te_t) > 10 and te_t.min() > tr_t.max():
        r_tr = len(tr_t) / max((tr_t.max() - tr_t.min()).total_seconds(), 1.0)
        r_te = len(te_t) / max((te_t.max() - te_t.min()).total_seconds(), 1.0)
        ratio = r_te / max(r_tr, 1e-12)
        if not (0.4 <= ratio <= 2.5):
            log(f"stream: пропуск из-за разной плотности train/test ({ratio:.2f}x)")
            return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)

    v_idx = np.flatnonzero(valid)
    nv = len(v_idx)
    ts_v = ts.iloc[v_idx]
    order = np.argsort(ts_v.values, kind="mergesort")
    orig_pos = v_idx[order]
    ts_ns = ts_v.values.astype("int64")[order]
    ts_s = ts_ns / 1e9
    # STRICTLY < t: searchsorted(..., side='left') excludes BOTH future events AND simultaneous ties!
    j_end = np.searchsorted(ts_ns, ts_ns, side="left")

    def _scatter(vals_ord):
        arr = np.full(n_tot, np.nan, dtype="float32")
        arr[orig_pos] = vals_ord.astype("float32")
        return arr

    out = {}
    for w_min in (15, 30, 60, 120, 360, 1440):
        j_beg = np.searchsorted(ts_ns, ts_ns - int(w_min * 60 * 1e9), side="left")
        out[f"stream__cnt_{w_min}m"] = _scatter(j_end - j_beg)
    for k in (1, 2, 3, 5, 10):
        idx_k = j_end - k
        gap = np.where(idx_k >= 0, (ts_s - ts_s[np.maximum(idx_k, 0)]) / 60.0, np.nan)
        out[f"stream__gap_prev{k}_min"] = _scatter(gap)

    hr_ns = ts_v.dt.floor("h").values.astype("int64")[order]
    prev_hr_ns = hr_ns - int(3600 * 1e9)
    j_hr = np.searchsorted(ts_ns, hr_ns, side="left")
    j_phr = np.searchsorted(ts_ns, prev_hr_ns, side="left")
    cnt_cur = (j_end - j_hr).astype("float32")
    cnt_prev = (j_hr - j_phr).astype("float32")
    min_in_hr = (ts_v.dt.minute.values + ts_v.dt.second.values / 60.0).astype("float32")[order]
    rate_cur = cnt_cur / (min_in_hr + 4.0) * 60.0
    out["stream__cnt_cur_hr_past"] = _scatter(cnt_cur)
    out["stream__cnt_prev_hr"] = _scatter(cnt_prev)
    out["stream__rate_cur_hr_past"] = _scatter(rate_cur)
    out["stream__rate_ratio_cur_prev"] = _scatter(rate_cur / (cnt_prev + 0.5))
    cnt_24h = (j_end - np.searchsorted(ts_ns, ts_ns - int(1440 * 60 * 1e9), side="left")).astype("float32")
    cnt_60m = (j_end - np.searchsorted(ts_ns, ts_ns - int(60 * 60 * 1e9), side="left")).astype("float32")
    out["stream__burst_vs_24h"] = _scatter(rate_cur / (cnt_24h / 24.0 + 0.25))
    out["stream__burst_60m_vs_24h"] = _scatter(cnt_60m / (cnt_24h / 24.0 + 0.25))

    # Strictly-past rolling sums, rates, and recency for raw binary (0/1) event flags (detected strictly on train)
    bin_cols = [c for c in raw_cols if c != tc and not str(c).lower().endswith("_id") and str(c).lower() != "id"
                and (pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c]))
                and train[c].dropna().nunique() == 2
                and set(pd.to_numeric(train[c].dropna(), errors="coerce").unique()).issubset({0, 1, 0.0, 1.0})][:8]
    for c in bin_cols:
        vals = pd.to_numeric(comb[c], errors="coerce").fillna(0.0).values[orig_pos].astype("float64")
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        # Vectorized O(N) time since latest positive event strictly < t (safe under timestamp ties!)
        pos_idx = np.where(vals >= 0.5, np.arange(nv, dtype=np.int64), -1)
        last_pos_le = np.maximum.accumulate(pos_idx)
        lp = np.where(j_end > 0, last_pos_le[np.maximum(j_end - 1, 0)], -1)
        gap_pos = np.where(lp >= 0, (ts_s - ts_s[np.maximum(lp, 0)]) / 60.0, np.nan)
        out[f"stream__{c}_gap_pos_min"] = _scatter(gap_pos)
        out[f"stream__{c}_pos_in_cur_hr"] = _scatter((lp >= j_hr).astype("float32"))
        for w_min in (30, 60, 180):
            j_beg = np.searchsorted(ts_ns, ts_ns - int(w_min * 60 * 1e9), side="left")
            s_w = csum[j_end] - csum[j_beg]
            c_w = np.maximum(j_end - j_beg, 1)
            out[f"stream__{c}_sum_{w_min}m"] = _scatter(s_w)
            out[f"stream__{c}_rate_{w_min}m"] = _scatter(s_w / c_w)
        out[f"stream__{c}_cur_hr_past"] = _scatter(csum[j_end] - csum[j_hr])
        out[f"stream__{c}_prev_hr"] = _scatter(csum[j_hr] - csum[j_phr])

    # Strictly-past rolling composition for low-cardinality raw categoricals (2..8 unique values)
    cat_cands = [c for c in raw_cols if c != tc and c not in bin_cols
                 and not str(c).lower().endswith("_id") and str(c).lower() != "id"
                 and not pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])
                 and 2 <= train[c].dropna().nunique() <= 8][:3]
    j_60 = np.searchsorted(ts_ns, ts_ns - int(60 * 60 * 1e9), side="left")
    for c in cat_cands:
        # Determine tracked category vocabulary strictly from train (never from future test rows).
        top_vals = train[c].astype("string").fillna("__nan__").value_counts().index[:5]
        s_ord = comb[c].astype("string").fillna("__nan__").values[orig_pos]
        for idx_v, v in enumerate(top_vals):
            ind = (s_ord == v).astype("float64")
            csum = np.concatenate([[0.0], np.cumsum(ind)])
            tag = re.sub(r"[^\w]+", "_", str(v)).strip("_")[:12] or f"v{idx_v}"
            out[f"stream__{c}_{tag}_cur_hr"] = _scatter(csum[j_end] - csum[j_hr])
            out[f"stream__{c}_{tag}_60m"] = _scatter(csum[j_end] - csum[j_60])

    # Strictly-past per-group event intensity (< t, exact int64 nanosecond lexsort: zero tie/precision leakage!)
    cat_set_done = set(cat_cands)
    grp_cands = []
    for c in raw_cols:
        if c == tc or c in bin_cols or c in cat_set_done:
            continue
        is_id = str(c).lower().endswith("_id")
        is_cat = not pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])
        if not (is_id or is_cat):
            continue
        nu = int(train[c].nunique(dropna=True))
        if (is_id and 2 <= nu <= 50_000) or (not is_id and 9 <= nu <= 50_000):
            grp_cands.append((1 if is_id else 0, -abs(np.log1p(nu) - np.log1p(50.0)), c, nu))
    grp_cands.sort(reverse=True)
    NS_SEC = 1_000_000_000
    ts_ns_v = ts_v.values.astype("int64")
    hr_ns_v = ts_v.dt.floor("h").values.astype("int64")
    w1h = 3600 * NS_SEC
    w24 = 86400 * NS_SEC
    w7d = 7 * 86400 * NS_SEC

    for _, _, gc, _ in grp_cands[:3]:
        if int(train[gc].notna().sum()) < 50:
            continue
        g_vals = comb[gc].iloc[v_idx].astype("string")
        codes_v, _ = pd.factorize(g_vals, sort=False)
        codes_v = codes_v.astype(np.int64)
        g_idx = np.flatnonzero(codes_v >= 0)
        if len(g_idx) == 0:
            continue
        # Exact int64 nanosecond lexsort per group (zero rounding, zero span_pad overflow)
        g_ord = np.lexsort((ts_ns_v[g_idx], codes_v[g_idx]))
        sel = g_idx[g_ord]
        gs_c = codes_v[sel]
        gs_t = ts_ns_v[sel]
        gs_hr = hr_ns_v[sel]
        g_pos = v_idx[sel]
        n_gs = len(sel)

        _, g_starts = np.unique(gs_c, return_index=True)
        g_ends = np.r_[g_starts[1:], n_gs]

        cnt_1h = np.zeros(n_gs, dtype=np.float32)
        cnt_h0 = np.zeros(n_gs, dtype=np.float32)
        cnt_24h = np.zeros(n_gs, dtype=np.float32)
        cnt_7d = np.zeros(n_gs, dtype=np.float32)
        g_gap_h = np.full(n_gs, np.nan, dtype=np.float32)

        for s, e in zip(g_starts.tolist(), g_ends.tolist()):
            gt = gs_t[s:e]
            gh = gs_hr[s:e]
            # STRICTLY < t to the exact nanosecond (excludes both future rows and simultaneous ties!)
            jc = np.searchsorted(gt, gt, side="left")
            j1h = np.searchsorted(gt, gt - w1h, side="left")
            j24 = np.searchsorted(gt, gt - w24, side="left")
            j7d = np.searchsorted(gt, gt - w7d, side="left")
            jh0 = np.searchsorted(gt, gh, side="left")
            cnt_1h[s:e] = (jc - j1h).astype(np.float32)
            cnt_h0[s:e] = (jc - jh0).astype(np.float32)
            cnt_24h[s:e] = (jc - j24).astype(np.float32)
            cnt_7d[s:e] = (jc - j7d).astype(np.float32)
            has_p = jc > 0
            if has_p.any():
                g_gap_h[s:e][has_p] = ((gt[has_p] - gt[jc[has_p] - 1]) / (3600.0 * NS_SEC)).astype(np.float32)

        def _scatter_g(arr_gs):
            arr = np.full(n_tot, np.nan, dtype="float32")
            arr[g_pos] = arr_gs.astype("float32")
            return arr

        out[f"stream_grp__{gc}__cnt_1h"] = _scatter_g(cnt_1h)
        out[f"stream_grp__{gc}__cnt_since_h"] = _scatter_g(cnt_h0)
        out[f"stream_grp__{gc}__cnt_24h"] = _scatter_g(cnt_24h)
        out[f"stream_grp__{gc}__cnt_7d"] = _scatter_g(cnt_7d)
        out[f"stream_grp__{gc}__gap_h"] = _scatter_g(g_gap_h)

    df = pd.DataFrame(out)
    log(f"строго причинный поток событий ({tc}, строго < t): +{df.shape[1]} признаков")
    return df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True)
