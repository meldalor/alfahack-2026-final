"""Быстрая проверка связки агента с LLM на настоящем API (2–5 минут, несколько запросов).

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://changellenge.up.railway.app/v1
    python tools/llm_check.py <каталог с task.md и данными>

Проверяет: ответ модели приходит и разбирается, спецификация задачи подтверждается, раунд гипотез
даёт применимые действия, код признаков от модели выполняется в отдельном процессе.
Работает на подвыборке строк, файлы задачи не меняет (всё пишется во временный каталог).
"""

import os
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ag_features import detect_time_split, prepare  # noqa: E402
from ag_llm import LLM  # noqa: E402
from ag_models import make_plan  # noqa: E402
from ag_research import Researcher  # noqa: E402
from ag_task import build_spec, find_files, llm_refine_spec, read_table  # noqa: E402
from ag_utils import Budget, effective_cpus  # noqa: E402


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("нужны OPENAI_API_KEY и OPENAI_BASE_URL")
    task_dir = Path(sys.argv[1]).resolve()
    task_path = task_dir / "task.md"
    text = task_path.read_text(encoding="utf-8")
    if "--cached" in sys.argv:
        # Explicitly reuse this run's extracted DB tables, never a neighbouring task.
        metadata = json.loads((task_dir / "agent_work" / "spec.json").read_text())
        files = metadata["files"]
    else:
        files = find_files(task_path, text)
    print("файлы:", files)
    train = read_table(Path(files["train"]))
    test = read_table(Path(files["test"]))
    sample = read_table(Path(files["sample"])) if "sample" in files else None
    dic = read_table(Path(files["dictionary"])) if "dictionary" in files else None
    if len(train) > 8000:
        train = train.sample(8000, random_state=0).reset_index(drop=True)
    if len(test) > 3000:
        test = test.head(3000)

    spec = build_spec(task_path, train, test, sample, dic, files)
    print(f"спецификация без LLM: {spec.task_type}, {spec.metric}, таргет {spec.target}")
    llm = LLM(max_calls=6, call_timeout=90, time_limit=360)
    hints = llm_refine_spec(spec, llm, train, test, sample)
    print("принято из ответа LLM:", {k: v for k, v in hints.items() if k != "raw"})
    print("идеи LLM:", (hints.get("raw") or {}).get("hypotheses"))

    prep = prepare(train, test, spec)
    tc = detect_time_split(prep.X, prep.Xt, prep.cat_cols)
    plan = make_plan(prep.X, prep.y, spec.task_type, time_col=tc, n_folds=3)
    work = Path(tempfile.mkdtemp(prefix="llm_check_"))
    r = Researcher(spec, prep, plan, llm, Budget(1800), effective_cpus(), work, llm_hints=hints.get("raw") or {})
    r.baseline()
    added = r.llm_round()
    print(f"раунд гипотез: добавлено {added}")
    proposed = sorted(list(r.queue), key=lambda h: h.get("kind") != "custom")
    for h in proposed[:3]:
        if h.get("source") == "llm":
            r.try_one(h)
    print("\nжурнал:")
    for e in r.journal:
        print(" ", e.get("id"), e.get("source"), e.get("title"), "->", e.get("score"), e.get("accepted"),
              e.get("error", "")[-200:] if e.get("error") else "")
    print("\nстатистика LLM:", llm.stats())
    evaluated = [e for e in r.journal if e.get("source") == "llm" and e.get("score") is not None and not e.get("error")]
    ok = llm.calls > 0 and llm.disabled_reason is None and added > 0 and bool(evaluated)
    if ok:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print("Артефакты для диагностики:", work)
    print("\nИТОГ:", "LLM работает" if ok else "есть проблемы — см. вывод выше")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
