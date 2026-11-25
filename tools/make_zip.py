"""Сборка архива для платформы: solution.py, все модули ag_*.py и requirements.txt в корне ZIP, без папок.

python tools/make_zip.py [путь к zip]   (по умолчанию dist/solution.zip)
python tools/make_zip.py dist/probe_cls.zip --probe-cls   диагностика: регрессия остаётся константой,
    поэтому балл на лидерборде = S_cls / 2 (показывает вклад классификации)
python tools/make_zip.py dist/solution_nollm.zip --no-llm   тот же агент без LLM (для сравнения с LLM)
python tools/make_zip.py dist/solution_noshift.zip --no-shift   без поправки на сдвиг долей классов (её вклад)
"""

import ast
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


PROBE_ANCHOR = "def solve_target(spec, train, test, llm, hints, budget, writer, work):\n    global N_JOBS\n"
PROBE_PATCH = ('    if spec.task_type == "regression":\n'
               '        log("ДИАГНОСТИКА: регрессия намеренно оставлена константой")\n'
               '        return\n')
NOLLM_ANCHOR = "def run(task_path: Path):\n"
NOLLM_PATCH = '    os.environ.pop("OPENAI_API_KEY", None)  # сборка без LLM\n'
NOSHIFT_ANCHOR = 'def label_shift(spec, prep, y, mask, oof, test):\n    """Оценка сдвига долей классов в тесте (для метрик по меткам); None, если неприменимо или не удалось."""\n'
NOSHIFT_PATCH = '    return None  # сборка без поправки на сдвиг долей классов\n'



def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    probe = "--probe-cls" in sys.argv
    nollm = "--no-llm" in sys.argv
    noshift = "--no-shift" in sys.argv
    out = Path(args[0]) if args else ROOT / "dist" / "solution.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "solution.py"] + sorted(ROOT.glob("ag_*.py")) + [ROOT / "requirements.txt"]
    for f in files:
        if not f.exists():
            raise SystemExit(f"нет файла {f}")
        if f.suffix == ".py":
            ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
    # все локальные импорты должны лежать в корне архива
    names = {f.stem for f in files if f.suffix == ".py"}
    for f in files:
        if f.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.startswith("ag_") and m not in names:
                    raise SystemExit(f"{f.name} импортирует отсутствующий модуль {m}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if (probe or nollm or noshift) and f.name == "solution.py":
                text = f.read_text(encoding="utf-8")
                for flag, anchor, patch in ((probe, PROBE_ANCHOR, PROBE_PATCH), (nollm, NOLLM_ANCHOR, NOLLM_PATCH),
                                            (noshift, NOSHIFT_ANCHOR, NOSHIFT_PATCH)):
                    if flag:
                        if anchor not in text:
                            raise SystemExit("не найдено место для правки в solution.py")
                        text = text.replace(anchor, anchor + patch, 1)
                z.writestr(f.name, text)
            else:
                z.write(f, arcname=f.name)
    print(f"{out}: {len(files)} файлов, {out.stat().st_size / 1024:.0f} КБ")
    for f in files:
        print("  ", f.name)


if __name__ == "__main__":
    main()
