"""Подготовка признаков и библиотека преобразований, из которых строятся гипотезы."""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ag_utils import log, log_exc

DATE_RE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?")


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
    both = pd.to_datetime(pd.concat([a, b], ignore_index=True), errors="coerce")
    out = pd.DataFrame(index=both.index)
    out[f"{name}__year"] = both.dt.year
    out[f"{name}__month"] = both.dt.month
    out[f"{name}__day"] = both.dt.day
    out[f"{name}__dow"] = both.dt.dayofweek
    out[f"{name}__doy"] = both.dt.dayofyear
    if (both.dt.hour.fillna(0) != 0).mean() > 0.05:
        out[f"{name}__hour"] = both.dt.hour
    out[f"{name}__ord"] = (both - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    out = out.astype("float32")
    return out.iloc[: len(a)].reset_index(drop=True), out.iloc[len(a):].reset_index(drop=True)


def prepare(train: pd.DataFrame, test: pd.DataFrame, spec) -> Prepared:
    train = train[train[spec.target].notna()].reset_index(drop=True)
    test = test.reset_index(drop=True)
    exclude = {spec.target, spec.id_col}
    feats = [c for c in test.columns if c in train.columns and c not in exclude]

    if spec.task_type == "regression":
        y = pd.to_numeric(train[spec.target], errors="coerce").astype(float).values
    else:
        mapping = {c: i for i, c in enumerate(spec.classes)}
        y = train[spec.target].map(mapping).values
        if pd.isna(y).any():
            # значения таргета не из списка классов (например, 1 и 1.0): сопоставляем по строке
            smap = {str(c): i for i, c in enumerate(spec.classes)}
            y = train[spec.target].astype(str).map(smap).values
        y = np.asarray(y, dtype=int)

    X_parts, T_parts = [], []
    cat_cols, text_cols, raw_text, dropped = [], [], {}, []
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
                X_parts.append(xa)
                T_parts.append(xb)
                continue
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                if name in cat_hint and pd.concat([a, b]).nunique() <= 1000 and _codes_seen(name, a, b):
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
            na = pd.to_numeric(sa, errors="coerce")
            nn = sa.notna().sum()
            if nn > 0 and na.notna().sum() >= 0.97 * nn and (name not in cat_hint or not _codes_seen(name, sa, sb)):
                X_parts.append(na.astype("float32").rename(name).to_frame())
                T_parts.append(pd.to_numeric(sb, errors="coerce").astype("float32").rename(name).to_frame())
                continue
            if _is_datetime_series(sa):
                xa, xb = _expand_datetime(name, sa, sb)
                X_parts.append(xa)
                T_parts.append(xb)
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

    # категориальные: единый словарь train+test; очень высокая кардинальность -> числовой код по частоте
    final_cats = []
    for c in cat_cols:
        both = pd.concat([X[c], Xt[c]], ignore_index=True).astype("string")
        vc = both.value_counts(dropna=True)
        if len(vc) > 1000:
            rank = {v: i for i, v in enumerate(vc.index)}
            freq = both.map(vc).astype("float32")
            code = both.map(rank).astype("float32")
            X[c] = code.iloc[: len(X)].values
            Xt[c] = code.iloc[len(X):].values
            X[c + "__freq"] = freq.iloc[: len(X)].values
            Xt[c + "__freq"] = freq.iloc[len(X):].values
        else:
            cats = pd.CategoricalDtype(categories=list(vc.index))
            X[c] = pd.Categorical(X[c].astype("string").astype(object), dtype=cats)
            Xt[c] = pd.Categorical(Xt[c].astype("string").astype(object), dtype=cats)
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
                 "text_cols": text_cols, "dropped": dropped,
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


def op_freq(X, Xt, cols):
    out_a, out_b = {}, {}
    for c in cols:
        both = pd.concat([X[c], Xt[c]], ignore_index=True)
        vc = both.astype("string").value_counts(dropna=False)
        f = both.astype("string").map(vc).astype("float32").values
        out_a[f"{c}__cnt"] = f[: len(X)]
        out_b[f"{c}__cnt"] = f[len(X):]
    return pd.DataFrame(out_a), pd.DataFrame(out_b)



def op_nan_count(X, Xt):
    return (pd.DataFrame({"row__nan": X.isna().sum(1).astype("float32").values}),
            pd.DataFrame({"row__nan": Xt.isna().sum(1).astype("float32").values}))


def op_dup_count(X, Xt):
    both = pd.concat([X, Xt], ignore_index=True)
    h = pd.util.hash_pandas_object(both.astype("string"), index=False)
    cnt = h.map(h.value_counts()).astype("float32").values
    return pd.DataFrame({"row__dup": cnt[: len(X)]}), pd.DataFrame({"row__dup": cnt[len(X):]})


def op_group_dev(X, Xt, pairs):
    out_a, out_b = {}, {}
    for cat, num in pairs:
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
    res = res.astype("float32")
    return res.iloc[:n].reset_index(drop=True), res.iloc[n:].reset_index(drop=True)


def entity_lag_cols(X, y, key_cols, t_train, exclude=(), max_cols: int = 4, min_corr: float = 0.15):
    """Числовые признаки, изменение которых у объекта между его периодами связано с изменением таргета
    (|corr(Δx, Δy)| по соседним наблюдениям одного объекта в train). Берутся только сильные: не слабее
    min_corr и половины лучшего. На WiDS: рейтинг ENERGY STAR (0.42) — да, погода (~0.1, 50 похожих колонок) —
    нет: с ней RMSE 34.90 против 34.67 без неё."""
    skip = set(key_cols) | set(exclude)
    cols = [c for c in X.columns if c not in skip and "__" not in c and pd.api.types.is_numeric_dtype(X[c])
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
        if c == time_col or "__" in c:
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

    static = [c for c, (p, nu) in stats.items() if p >= 0.6]
    cands = []
    if extra:
        ex = [c for c in extra if c in X.columns and c != time_col]
        if ex:
            cands.append(ex)
    # жадно: добавляем колонки, пока растёт uniq*match
    order = sorted(static, key=lambda c: -stats[c][1])
    cur, cur_q = [], 0.0
    for c in order:
        u, m = quality(cur + [c])
        q = u * m
        if q > cur_q + 1e-4:
            cur, cur_q = cur + [c], q
    if cur:
        cands.append(cur)
        # тот же ключ, уточнённый стабильными категориями: реже склеивает разные объекты
        cats_static = sorted([c for c in static if c in set(cat_cols) and c not in cur and stats[c][0] >= 0.9],
                             key=lambda c: -stats[c][1])
        if cats_static:
            cands.append(cur + cats_static[:1])
        if len(cats_static) > 1:
            cands.append(cur + cats_static)
    if len(static) > 1:
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
    groups = [c for c, (p, nu) in stats.items() if p >= 0.8 and 10 <= nu <= 0.5 * per_rows]
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
        both = pd.concat([sa, sb], ignore_index=True).astype(str)
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
        if periods is not None:
            per = np.asarray(periods, dtype=float)
            if len(per) != len(y):
                raise ValueError("Text periods must align with target rows")
            folds = [(np.flatnonzero(per < value), np.flatnonzero(per == value))
                     for value in np.sort(np.unique(per[np.isfinite(per)]))]

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
    for nm in null_markers:
        prefix = nm[:-len("__raw_nulls")] + "__"
        t_cols = [c for c in num_all if c.startswith(prefix) and c != nm and c != prefix + "matched"]
        other_num = [c for c in num_all if not c.startswith(prefix)]
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
