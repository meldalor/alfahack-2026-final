"""Понимание задачи: чтение task.md, поиск файлов, определение таргета, id, типа задачи и метрики."""

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ag_metrics import METRICS, PROBA_METRICS
from ag_utils import log, log_exc, short

DATA_EXT = (".csv", ".tsv", ".parquet", ".pq", ".feather", ".json", ".jsonl", ".txt")
SKIP_DIRS = {"agent_work", "__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}


@dataclass
class TaskSpec:
    task_path: str
    text: str
    files: dict = field(default_factory=dict)
    id_col: str | None = None
    target: str = ""
    task_type: str = "regression"  # regression | binary | multiclass
    metric: str = "rmse"
    metric_source: str = "default"
    output_kind: str = "value"  # value | label | proba | proba_multi
    sub_columns: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    positive_label: object = None
    cat_hint: list = field(default_factory=list)
    descriptions: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    extra_targets: list = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("text", None)
        d["classes"] = [str(c) for c in self.classes][:50]
        return d


# ---------------------------------------------------------------- чтение таблиц


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    if suf in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=suf == ".jsonl")
        except ValueError:
            return pd.read_json(path, lines=True)
    sep = "\t" if suf == ".tsv" else ","
    enc = "utf-8-sig"  # снимает BOM, если файл сохранён из Excel; без BOM работает как utf-8
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False, encoding=enc)
    except UnicodeDecodeError:
        enc = "cp1251"
        df = pd.read_csv(path, sep=sep, low_memory=False, encoding=enc)
    if df.shape[1] == 1:
        head = str(df.columns[0])
        for alt in (";", "\t", "|"):
            if alt in head:
                return pd.read_csv(path, sep=alt, low_memory=False, encoding=enc)
    return df


# ---------------------------------------------------------------- поиск файлов


def _walk(base: Path, max_depth: int):
    out = []
    base = base.resolve()
    for root, dirs, files in os.walk(base):
        rel = Path(root).relative_to(base)
        depth = len(rel.parts)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        if depth >= max_depth:
            dirs[:] = []
        for f in files:
            if f.lower().endswith(DATA_EXT):
                out.append(Path(root) / f)
    return out


def _role_of(name: str):
    stem = Path(name.lower()).stem
    if re.search(r"sample|template|example", stem):
        return "sample"
    if re.search(r"dictionary|dict|description|schema|columns|meta", stem):
        return "dictionary"
    # граница слова слева: «constraints» или «latest» не должны считаться train/test
    if re.search(r"(^|[^a-z])train", stem):
        return "train"
    if re.search(r"(^|[^a-z])test", stem):
        return "test"
    return None


def find_files(task_path: Path, text: str, search_parent: bool = True) -> dict:
    base = task_path.parent.resolve()
    own_output = (base / "submission.csv").resolve()
    found = {}

    # 1) пути, упомянутые в task.md (относительно каталога task.md, затем по имени файла)
    mentioned = re.findall(r"[\w./\\\-]+\.(?:csv|tsv|parquet|pq|feather|jsonl?|txt)", text)
    candidates = _walk(base, 6)
    if search_parent and not any(_role_of(p.name) == "train" for p in candidates):
        candidates += _walk(base.parent, 4)
    by_name = {}
    for p in candidates:
        by_name.setdefault(p.name.lower(), []).append(p)

    def resolve(m: str):
        m = m.strip().strip("`'\"").replace("\\", "/")
        for root in (base, base.parent, Path.cwd()):
            p = (root / m).resolve()
            if p.is_file():
                return p
        lst = by_name.get(Path(m).name.lower())
        if lst:
            # если имя файла неоднозначно, выбираем путь с наибольшим числом совпадающих компонентов
            parts = [x.lower() for x in Path(m).parts]
            return max(lst, key=lambda q: (sum(x.lower() in parts for x in q.parts), -len(q.parts)))
        return None

    for m in mentioned:
        p = resolve(m)
        if p is None or p == own_output:
            continue
        role = _role_of(p.name)
        if role and role not in found:
            found[role] = p

    # 2) поиск по дереву каталогов
    for p in sorted(candidates, key=lambda q: (len(q.parts), str(q))):
        if p.resolve() == own_output or "agent_work" in p.parts:
            continue
        role = _role_of(p.name)
        if role and role not in found:
            found[role] = p

    # тест и обучение обычно лежат рядом: если нашли в разных местах, ищем пару в каталоге train
    if "train" in found:
        tdir = found["train"].parent
        for role in ("test", "sample", "dictionary"):
            near = [q for q in tdir.iterdir() if q.is_file() and _role_of(q.name) == role] if tdir.is_dir() else []
            if near and (role not in found or found[role].parent != tdir):
                found[role] = sorted(near, key=lambda q: len(q.name))[0]
    return {k: str(v) for k, v in found.items()}


# ---------------------------------------------------------------- метрика и тип задачи

METRIC_PATTERNS = [
    ("rmsle", r"rmsle|root mean squared log|root mean square log|msle"),
    ("rmse", r"\brmse\b|root mean squared? error|корень из средне"),
    ("mse", r"\bmse\b|mean squared error|среднеквадратичн"),
    ("mae", r"\bmae\b|mean absolute error|средняя абсолютная ошибка|\bwape\b|\bwmape\b"),
    ("smape", r"smape"),
    ("mape", r"\bmape\b|mean absolute percentage|средняя абсолютная процентная"),
    ("medae", r"median absolute error|medae"),
    ("r2", r"\br2\b|\br\^2|\br²|r-squared|коэффициент детерминации|r2_score"),
    ("f1_macro", r"macro[\s\-_]*f1|f1[\s\-_]*macro|average\s*=\s*['\"]macro['\"]|макро[\s\-]*f1|f1[\s\-]*макро"),
    ("f1_weighted", r"weighted[\s\-_]*f1|f1[\s\-_]*weighted|average\s*=\s*['\"]weighted['\"]"),
    ("f1_micro", r"micro[\s\-_]*f1|f1[\s\-_]*micro|average\s*=\s*['\"]micro['\"]"),
    ("balanced_accuracy", r"balanced[\s\-_]*accuracy|сбалансированн\w* точност"),
    ("mcc", r"\bmcc\b|matthews"),
    ("qwk", r"quadratic weighted kappa|\bqwk\b|cohen.{0,10}kappa|каппа"),
    ("average_precision", r"pr[\s\-_]*auc|auc[\s\-_]*pr|average[\s\-_]*precision|precision[\s\-_]*recall[\s\-_]*auc"),
    ("roc_auc", r"roc[\s\-_]*auc|\bauc[\s\-_]*roc|\bauc\b|\bgini\b|джини"),
    ("logloss", r"log[\s\-_]*loss|cross[\s\-_]*entropy|кросс[\s\-]*энтропи|логлосс|brier"),
    ("f1", r"\bf1\b|f1[\s\-_]*score|f-?мера"),
    ("accuracy", r"\baccuracy\b|доля верных|точность классификации"),
]


def _first_metric(scope: str):
    hits = []
    for name, pat in METRIC_PATTERNS:
        mm = re.search(pat, scope)
        if mm:
            hits.append((mm.start(), -(mm.end() - mm.start()), name))
    return sorted(hits)[0][2] if hits else None


def parse_metric(text: str):
    """Ищет основную метрику: сначала после «основная метрика», затем «метрика оценки»/«metric», затем по всему тексту."""
    low = text.lower()
    groups = [r"основн\w*\s+метрик", r"метрик\w*\s+(?:оценк|качеств|соревнован|задач)", r"метрик|metric",
              r"оценива|evaluat|scor"]
    for prio, pat in enumerate(groups):
        for m in re.finditer(pat, low):
            found = _first_metric(low[m.start() : m.start() + 200])
            if found:
                return found, "explicit" if prio < 3 else "text"
    found = _first_metric(low)
    return (found, "text") if found else (None, "none")


def detect_task_type(text: str, y: pd.Series, metric: str | None) -> str:
    low = text.lower()
    yy = y.dropna()
    nun = yy.nunique()
    numeric = pd.api.types.is_numeric_dtype(yy) and not pd.api.types.is_bool_dtype(yy)
    if not numeric and not pd.api.types.is_bool_dtype(yy) and len(yy):
        from ag_features import _parse_dirty_numeric

        parsed, nn = _parse_dirty_numeric(yy)
        if nn > 0 and parsed.notna().sum() >= 0.85 * nn:
            yy_num = parsed.dropna()
            if len(yy_num):
                yy = yy_num
                nun = int(yy.nunique())
                numeric = True
    says_reg = bool(re.search(r"регресс|regression", low))
    says_cls = bool(re.search(r"классификац|classification|classif", low))
    says_bin = bool(re.search(r"бинарн|binary", low))
    if metric and METRICS.get(metric, (None, None))[1] == "regression" and numeric:
        return "regression"
    if not numeric:
        return "binary" if nun == 2 else "multiclass"
    integer_like = bool(np.all(np.isclose(yy.values.astype(float), np.round(yy.values.astype(float)))))
    if says_reg and not says_cls:
        return "regression"
    if metric and METRICS.get(metric, (None, None))[1] == "classification":
        return "binary" if nun == 2 else "multiclass"
    if says_cls or says_bin:
        if integer_like or nun <= 50:
            return "binary" if nun == 2 else "multiclass"
    if nun == 2:
        return "binary"
    if integer_like and nun <= 20:
        return "multiclass"
    return "regression"


POSITIVE_WORDS = {"1", "1.0", "yes", "true", "y", "positive", "pos", "fraud", "churn", "default", "spam", "attack",
                  "bad", "да", "истина", "t"}


def choose_positive(classes: list, y: pd.Series):
    """Положительный класс для вероятностного ответа. Для чисел — больший (как pos_label в sklearn
    по умолчанию: {0,1} -> 1, {1,2} -> 2), для строк — по словарю «yes/true/fraud…», иначе больший."""
    if len(classes) != 2:
        return None
    if all(isinstance(c, (int, float, np.integer, np.floating, bool, np.bool_)) for c in classes):
        return max(classes)
    for c in classes:
        if str(c).strip().lower() in POSITIVE_WORDS:
            return c
    try:
        return max(classes)
    except Exception:
        counts = y.value_counts()
        return counts.index[-1]


# ---------------------------------------------------------------- спецификация


def _dictionary_roles(dic: pd.DataFrame):
    roles, dtypes, desc = {}, {}, {}
    if dic is None or dic.empty:
        return roles, dtypes, desc
    cols = {c.lower(): c for c in dic.columns}
    name_col = next((cols[c] for c in ("feature", "column", "name", "field", "признак", "колонка", "variable") if c in cols),
                    dic.columns[0])
    role_col = next((cols[c] for c in ("role", "роль", "type_role") if c in cols), None)
    dtype_col = next((cols[c] for c in ("dtype", "type", "тип", "data_type") if c in cols), None)
    desc_col = next((cols[c] for c in cols if "descr" in c or "опис" in c), None)
    for _, row in dic.iterrows():
        name = str(row[name_col]).strip()
        if role_col is not None:
            roles[name] = str(row[role_col]).strip().lower()
        if dtype_col is not None:
            dtypes[name] = str(row[dtype_col]).strip().lower()
        if desc_col is not None:
            desc[name] = str(row[desc_col])[:200]
    return roles, dtypes, desc


def build_spec(task_path: Path, train: pd.DataFrame, test: pd.DataFrame, sample, dic, files: dict,
               force_target: str | None = None) -> TaskSpec:
    text = task_path.read_text(encoding="utf-8", errors="replace")
    spec = TaskSpec(task_path=str(task_path), text=text, files=files)
    roles, dtypes, desc = _dictionary_roles(dic)
    spec.descriptions = desc

    tr_cols, te_cols = list(train.columns), list(test.columns)
    only_train = [c for c in tr_cols if c not in te_cols]
    sample_cols = list(sample.columns) if sample is not None else []

    # id: первая колонка шаблона, роль в словаре, типичное имя
    id_col = None
    if sample_cols and sample_cols[0] in te_cols:
        id_col = sample_cols[0]
    if id_col is None:
        for c, r in roles.items():
            if ("id" == r or "identifier" in r or "идентиф" in r) and c in te_cols:
                id_col = c
                break
    if id_col is None:
        for c in te_cols:
            if c.lower() in ("id", "row_id", "index", "sample_id", "object_id") and test[c].is_unique:
                id_col = c
                break
    if id_col is None:
        mentioned = re.search(r"(?:\bID|идентификатор)\s*[:=—-]\s*[`\"]?([\w]+)", text, re.I)
        if mentioned and mentioned.group(1) in te_cols:
            c = mentioned.group(1)
            if test[c].is_unique and test[c].notna().all():
                id_col = c
    if id_col is None:
        candidates = [c for c in te_cols if c.lower().endswith("_id")
                      and test[c].is_unique and test[c].notna().all()
                      and c in train and train[c].is_unique]
        if len(candidates) == 1:
            id_col = candidates[0]
    spec.id_col = id_col

    # таргет: колонка шаблона, которой нет в тесте; роль в словаре; колонка только из train
    target = force_target
    for c in ([] if target else sample_cols):
        if c != id_col and c in tr_cols and c not in te_cols:
            target = c
            break
    if target is None:
        for c, r in roles.items():
            if ("target" in r or "таргет" in r or "целев" in r) and c in tr_cols:
                target = c
                break
    if target is None:
        cand = [c for c in only_train if c != id_col]
        if len(cand) == 1:
            target = cand[0]
        elif cand:
            low = text.lower()
            mentioned = [c for c in cand if c.lower() in low]
            target = (mentioned or cand)[-1]
    if target is None and len(sample_cols) >= 2 and sample_cols[1] in tr_cols:
        target = sample_cols[1]
    if target is None:
        raise RuntimeError("не удалось определить целевую переменную")
    spec.target = target

    metric, src = parse_metric(text)
    y = train[target]
    spec.task_type = detect_task_type(text, y, metric)
    if metric is None or METRICS[metric][1] != ("regression" if spec.task_type == "regression" else "classification"):
        if metric is not None:
            spec.notes.append(f"метрика {metric} не подходит к типу {spec.task_type}, берём по умолчанию")
        metric, src = ("rmse" if spec.task_type == "regression" else
                       ("f1_macro" if spec.task_type == "multiclass" else "f1")), "default"
    if metric == "f1" and spec.task_type == "multiclass":
        metric = "f1_macro"
    spec.metric, spec.metric_source = metric, src

    if spec.task_type != "regression":
        classes = sorted(y.dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        spec.classes = classes
        spec.positive_label = choose_positive(classes, y)
        if spec.task_type == "binary" and spec.positive_label is not None and classes[1] != spec.positive_label:
            spec.classes = [c for c in classes if c != spec.positive_label] + [spec.positive_label]

    # формат вывода по шаблону
    if spec.task_type == "regression":
        spec.output_kind = "value"
    else:
        def _canon_cls(v) -> str:
            if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
                return str(int(round(float(v))))
            return str(v).strip().lower()
        class_strs = {_canon_cls(k) for k in spec.classes}
        def _matches_class_col(col):
            s = str(col).strip().lower()
            if s in class_strs:
                return True
            for pfx in ("class_", "prob_", "proba_", "pred_", "p_", "target_", "label_"):
                if s.startswith(pfx) and s[len(pfx):] in class_strs:
                    return True
            return False
        class_cols = [c for c in sample_cols if c != id_col and _matches_class_col(c)]
        if len(class_cols) >= 2:
            spec.output_kind = "proba_multi"
        elif spec.metric in PROBA_METRICS:
            spec.output_kind = "proba" if spec.task_type == "binary" else "label"
            if spec.task_type == "multiclass":
                spec.notes.append("метрика по вероятностям, но шаблон с одной колонкой: выводим метки")
        else:
            spec.output_kind = "label"
    if sample_cols:
        spec.sub_columns = sample_cols
    else:
        spec.sub_columns = ([id_col] if id_col else []) + [target]
    # несколько таргетов в шаблоне (например, id,t1,t2): остальные решаются отдельно
    if spec.output_kind != "proba_multi":
        spec.extra_targets = [c for c in sample_cols if c not in (id_col, target) and c in tr_cols and c not in te_cols]

    spec.cat_hint = [c for c, t in dtypes.items() if c in te_cols and any(k in t for k in ("cat", "str", "object", "bool"))]
    return spec


def llm_refine_spec(spec: TaskSpec, llm, train: pd.DataFrame, test: pd.DataFrame, sample) -> dict:
    """LLM читает условие и подтверждает/уточняет спецификацию. Принимаются только проверяемые поля."""
    if llm is None or not llm.enabled:
        return {}
    cols = ", ".join(map(str, train.columns[:120]))
    sys_prompt = ("You are a senior data scientist. Read an ML task statement (often in Russian) and extract a precise "
                  "machine-readable specification. Reply with a single JSON object only.")
    user = f"""TASK STATEMENT (task.md):
-----
{spec.text[:9000]}
-----
train.csv columns: {cols}
first rows of train.csv:
{short(train.head(3).to_csv(index=False), 2500)}
columns only in train (not in test): {[c for c in train.columns if c not in test.columns]}
sample_submission columns: {list(sample.columns) if sample is not None else None}
Automatic guess: target={spec.target!r}, id={spec.id_col!r}, task_type={spec.task_type}, metric={spec.metric} (source={spec.metric_source}).

Return JSON with keys:
"target": target column name,
"id": id column name or null,
"task_type": one of "regression","binary","multiclass",
"metric": main metric, one of {sorted(METRICS)},
"positive_label": for binary tasks the positive class value if stated, else null,
"time_column": column that orders observations in time if the split is temporal, else null,
"entity_columns": list of columns that jointly identify the same real-world object across rows/periods (may be empty),
"categorical_columns": columns that are categorical codes even if numeric,
"important_notes": list of short facts from the statement that matter for modelling (units, leakage warnings, what NOT to use),
"hypotheses": list of 3-6 concrete ideas likely to improve the metric for THIS data."""
    obj = llm.ask_json(sys_prompt, user, max_tokens=5000, temperature=0.2)
    if not isinstance(obj, dict):
        return {}
    log("LLM-спецификация:", short(obj, 900))
    accepted = {}
    t = obj.get("target")
    if isinstance(t, str) and t in train.columns and t not in test.columns and t != spec.target:
        if spec.target in test.columns or spec.target not in train.columns:
            spec.target = t
            accepted["target"] = t
    m = obj.get("metric")
    if isinstance(m, str):
        m = m.strip().lower()
        task_ok = METRICS.get(m, (None, None))[1] == ("regression" if spec.task_type == "regression" else "classification")
        if m in METRICS and task_ok and spec.metric_source in ("default", "text") and m != spec.metric:
            spec.notes.append(f"метрика уточнена LLM: {spec.metric} -> {m}")
            spec.metric, spec.metric_source = m, "llm"
            accepted["metric"] = m
    pl = obj.get("positive_label")
    if spec.task_type == "binary" and pl is not None:
        match = [c for c in spec.classes if str(c) == str(pl)]
        # принимаем, только если такое значение действительно упомянуто в условии (защита от выдумки)
        if match and re.search(r"(?<![\w.])" + re.escape(str(match[0])) + r"(?![\w.])", spec.text):
            spec.positive_label = match[0]
            spec.classes = [c for c in spec.classes if c != match[0]] + [match[0]]
            accepted["positive_label"] = str(match[0])
    cats = obj.get("categorical_columns")
    if isinstance(cats, list):
        extra = [c for c in cats if isinstance(c, str) and c in test.columns and c not in spec.cat_hint]
        spec.cat_hint += extra
    notes = obj.get("important_notes")
    if isinstance(notes, list):
        spec.notes += [str(n)[:300] for n in notes[:8]]
    accepted["raw"] = {k: obj.get(k) for k in ("time_column", "entity_columns", "hypotheses")}
    return accepted
