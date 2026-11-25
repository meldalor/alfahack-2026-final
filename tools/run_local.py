"""Локальный прогон как на платформе: копирует корневые .py и requirements.txt в каталог задачи
и запускает `python solution.py task.md` из него.

python tools/run_local.py <каталог с task.md и данными> <рабочий каталог> [--nested]
--nested раскладывает данные по путям, указанным в task.md (как на платформе).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    nested = "--nested" in sys.argv
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    text = (src / "task.md").read_text(encoding="utf-8")
    shutil.copy(src / "task.md", dst / "task.md")
    for f in src.iterdir():
        if f.suffix != ".csv":
            continue
        target = dst / f.name
        if nested:
            m = re.search(r"[\w./-]*" + re.escape(f.name), text)
            if m:
                target = dst / m.group(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(f)
        except OSError:
            shutil.copy(f, target)
    for f in ROOT.glob("*.py"):
        shutil.copy(f, dst / f.name)
    shutil.copy(ROOT / "requirements.txt", dst / "requirements.txt")
    sys.exit(subprocess.call([sys.executable, "solution.py", "task.md"], cwd=dst))


if __name__ == "__main__":
    main()
