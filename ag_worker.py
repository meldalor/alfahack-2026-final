"""Отдельный процесс для кода признаков, сгенерированного LLM.

Запуск: python ag_worker.py <каталог эксперимента> [time_col].
В каталоге лежат code.py, train.pkl, test.pkl.
Код должен определить make_features(train, test) -> (new_train, new_test). Ошибка или зависание
этого кода не затрагивают основной процесс агента. При передаче time_col или AG_CHECK_TEMPORAL_LEAK=1
включается автоматический двухступенчатый poison-фильтр, отбрасывающий признаки с утечкой из будущих строк test.
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _extract_valid(a, b, train_cols, n_tr, n_te, keep_cols=None):
    if isinstance(a, pd.Series):
        a = a.to_frame()
    if isinstance(b, pd.Series):
        b = b.to_frame()
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        raise RuntimeError("make_features должна вернуть два DataFrame")
    if len(a) != n_tr or len(b) != n_te:
        raise RuntimeError(f"неверное число строк: {len(a)}/{n_tr}, {len(b)}/{n_te}")
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    cols_iter = keep_cols if keep_cols is not None else list(a.columns)
    out_a, out_b, keep = {}, {}, []
    for c in cols_iter:
        if c not in a.columns or c not in b.columns or c in train_cols:
            continue
        sa = pd.to_numeric(a[c], errors="coerce") if a[c].dtype == object else a[c]
        sb = pd.to_numeric(b[c], errors="coerce") if b[c].dtype == object else b[c]
        if pd.api.types.is_bool_dtype(sa):
            sa, sb = sa.astype(float), sb.astype(float)
        if not pd.api.types.is_numeric_dtype(sa) or not pd.api.types.is_numeric_dtype(sb):
            continue
        sa = sa.astype("float32").replace([np.inf, -np.inf], np.nan)
        sb = sb.astype("float32").replace([np.inf, -np.inf], np.nan)
        if keep_cols is None and sa.nunique() <= 1:
            continue
        out_a[c], out_b[c] = sa, sb
        keep.append(c)
    return pd.DataFrame(out_a), pd.DataFrame(out_b), keep


def _poison_future_test(test: pd.DataFrame, future_mask: np.ndarray, time_col: str | None) -> pd.DataFrame:
    """Отравляет все колонки будущих строк test (кроме time_col), чтобы выявить любые утечки будущего."""
    tp = test.copy()
    for col in tp.columns:
        if time_col and col == time_col:
            continue
        s = tp[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if "__POISON__" not in s.cat.categories:
                s = s.cat.add_categories(["__POISON__"])
            s = s.copy()
            s.iloc[future_mask] = "__POISON__"
            tp[col] = s
        elif pd.api.types.is_bool_dtype(s):
            tp.loc[future_mask, col] = ~s.loc[future_mask].fillna(False)
        elif pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s.loc[future_mask], errors="coerce").to_numpy(dtype=np.float64)
            uniq = set(np.round(vals[np.isfinite(vals)], 6).tolist())
            if uniq and uniq.issubset({0.0, 1.0}):
                new_v = 1.0 - np.nan_to_num(vals, nan=0.0)
            else:
                new_v = np.where(np.isfinite(vals), -3.7 * vals + 98765.43, 987654.0)
            if pd.api.types.is_integer_dtype(s.dtype):
                tp.loc[future_mask, col] = np.round(new_v).astype(s.dtype)
            else:
                tp.loc[future_mask, col] = new_v.astype(s.dtype if pd.api.types.is_float_dtype(s.dtype) else np.float32)
        else:
            tp.loc[future_mask, col] = "__POISON__"
    return tp


def _poison_filter(fn, train: pd.DataFrame, test: pd.DataFrame, a_df: pd.DataFrame, b_df: pd.DataFrame,
                   keep: list, time_col: str | None, first_sec: float) -> tuple[list, list]:
    """Двухступенчатый poison-фильтр для временных сплитов:
    1) Отравление значений будущих строк test (>= медианы времени или второй половины test)
    2) Полное удаление будущих строк test (онлайн-усечение, если быстрый расчёт)
    Любой признак, изменившийся в train или в ранней половине test, отбрасывается.
    """
    if len(test) < 10:
        return keep, []
    early_mask = None
    if time_col and time_col in test.columns:
        t_te = pd.to_numeric(test[time_col], errors="coerce").to_numpy(dtype=np.float64)
        fin = t_te[np.isfinite(t_te)]
        if len(fin) >= 6:
            t_cut = float(np.median(fin))
            early_mask = np.isfinite(t_te) & (t_te < t_cut)
            if int(early_mask.sum()) < 3:
                t_cut = float(np.quantile(fin, 0.75))
                early_mask = np.isfinite(t_te) & (t_te < t_cut)
            if int(early_mask.sum()) < 3:
                t_min = float(np.min(fin))
                early_mask = np.isfinite(t_te) & (t_te <= t_min)
    if early_mask is None:
        mid = len(test) // 2
        early_mask = np.arange(len(test)) < mid
    future_mask = ~early_mask
    if int(early_mask.sum()) < 2 or int(future_mask.sum()) < 2:
        return keep, []

    train_cols = set(train.columns)
    checks = []

    # Проверка A: отравление значений в будущих строках test
    try:
        np.random.seed(0)
        tp = _poison_future_test(test, future_mask, time_col)
        ap_raw, bp_raw = fn(train.copy(), tp)
        ap, bp, _ = _extract_valid(ap_raw, bp_raw, train_cols, len(train), len(test), keep_cols=keep)
        checks.append(("poison", ap, bp.loc[early_mask].reset_index(drop=True)))
    except Exception:
        pass

    # Проверка B: полное удаление будущих строк test (ловит зависимости от len(test), value_counts, квантилей)
    if first_sec < 15.0:
        try:
            np.random.seed(0)
            t_early = test.loc[early_mask].copy().reset_index(drop=True)
            at_raw, bt_raw = fn(train.copy(), t_early)
            at, bt, _ = _extract_valid(at_raw, bt_raw, train_cols, len(train), len(t_early), keep_cols=keep)
            checks.append(("trunc", at, bt.reset_index(drop=True)))
        except Exception:
            pass

    if not checks:
        return keep, []

    clean, leaky = [], []
    b_early_ref = b_df.loc[early_mask].reset_index(drop=True)
    for c in keep:
        is_bad = False
        va_ref = a_df[c].to_numpy(dtype=np.float64)
        vb_ref = b_early_ref[c].to_numpy(dtype=np.float64)
        for _, a_chk, b_chk in checks:
            if c not in a_chk.columns or c not in b_chk.columns:
                is_bad = True
                break
            va_c = a_chk[c].to_numpy(dtype=np.float64)
            vb_c = b_chk[c].to_numpy(dtype=np.float64)
            if not (np.allclose(va_ref, va_c, rtol=1e-4, atol=1e-4, equal_nan=True)
                    and np.allclose(vb_ref, vb_c, rtol=1e-4, atol=1e-4, equal_nan=True)):
                is_bad = True
                break
        if is_bad:
            leaky.append(c)
        else:
            clean.append(c)
    return clean, leaky


def main(work: Path, time_col: str | None = None):
    code = (work / "code.py").read_text(encoding="utf-8")
    train = pd.read_pickle(work / "train.pkl")
    test = pd.read_pickle(work / "test.pkl")
    scope = {"pd": pd, "np": np, "__name__": "llm_features"}
    exec(compile(code, "llm_features.py", "exec"), scope)
    fn = scope.get("make_features")
    if fn is None:
        raise RuntimeError("в коде нет функции make_features(train, test)")
    t0 = time.time()
    np.random.seed(0)
    a_raw, b_raw = fn(train.copy(), test.copy())
    first_sec = time.time() - t0
    a, b, keep = _extract_valid(a_raw, b_raw, set(train.columns), len(train), len(test))
    if not keep:
        raise RuntimeError("код не создал ни одного нового числового признака")
    keep = keep[:40]
    if time_col or os.environ.get("AG_CHECK_TEMPORAL_LEAK") == "1":
        keep, leaky = _poison_filter(fn, train, test, a, b, keep, time_col, first_sec)
        if leaky:
            print("POISON_DROP:", ", ".join(map(str, leaky)))
        if not keep:
            raise RuntimeError(
                "temporal leakage detected in LLM make_features (poison-filter dropped all features): "
                + ", ".join(map(str, leaky[:12]))
            )
    a[keep].to_pickle(work / "out_train.pkl")
    b[keep].to_pickle(work / "out_test.pkl")
    print("OK", len(keep), "признаков:", ", ".join(map(str, keep)))


if __name__ == "__main__":
    try:
        tcol = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
        main(Path(sys.argv[1]), time_col=tcol)
    except Exception:
        traceback.print_exc(limit=5)
        sys.exit(1)
