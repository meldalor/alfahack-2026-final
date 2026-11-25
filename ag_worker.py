"""Отдельный процесс для кода признаков, сгенерированного LLM.

Запуск: python ag_worker.py <каталог эксперимента>. В каталоге лежат code.py, train.pkl, test.pkl.
Код должен определить make_features(train, test) -> (new_train, new_test). Ошибка или зависание
этого кода не затрагивают основной процесс агента.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def main(work: Path):
    code = (work / "code.py").read_text(encoding="utf-8")
    train = pd.read_pickle(work / "train.pkl")
    test = pd.read_pickle(work / "test.pkl")
    scope = {"pd": pd, "np": np, "__name__": "llm_features"}
    exec(compile(code, "llm_features.py", "exec"), scope)
    fn = scope.get("make_features")
    if fn is None:
        raise RuntimeError("в коде нет функции make_features(train, test)")
    a, b = fn(train.copy(), test.copy())
    if isinstance(a, pd.Series):
        a = a.to_frame()
    if isinstance(b, pd.Series):
        b = b.to_frame()
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        raise RuntimeError("make_features должна вернуть два DataFrame")
    if len(a) != len(train) or len(b) != len(test):
        raise RuntimeError(f"неверное число строк: {len(a)}/{len(train)}, {len(b)}/{len(test)}")
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    keep = []
    for c in a.columns:
        if c not in b.columns or c in train.columns:
            continue
        sa = pd.to_numeric(a[c], errors="coerce") if a[c].dtype == object else a[c]
        sb = pd.to_numeric(b[c], errors="coerce") if b[c].dtype == object else b[c]
        if pd.api.types.is_bool_dtype(sa):
            sa, sb = sa.astype(float), sb.astype(float)
        if not pd.api.types.is_numeric_dtype(sa) or not pd.api.types.is_numeric_dtype(sb):
            continue
        sa = sa.astype("float32").replace([np.inf, -np.inf], np.nan)
        sb = sb.astype("float32").replace([np.inf, -np.inf], np.nan)
        if sa.nunique() <= 1:
            continue
        a[c], b[c] = sa, sb
        keep.append(c)
    if not keep:
        raise RuntimeError("код не создал ни одного нового числового признака")
    keep = keep[:40]
    a[keep].to_pickle(work / "out_train.pkl")
    b[keep].to_pickle(work / "out_test.pkl")
    print("OK", len(keep), "признаков:", ", ".join(map(str, keep)))


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception:
        traceback.print_exc(limit=5)
        sys.exit(1)
