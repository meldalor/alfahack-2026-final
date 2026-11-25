"""Общие утилиты агента: журнал, бюджет времени, ресурсы, безопасная запись файлов."""

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

START_TIME = time.time()


def elapsed() -> float:
    return time.time() - START_TIME


_log_lock = threading.Lock()
_log_file = None


def setup_log(path: Path):
    global _log_file
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(path, "a", encoding="utf-8")
    except Exception:
        _log_file = None


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    line = f"[{elapsed():7.1f}s] {msg}"
    with _log_lock:
        try:
            print(line, flush=True)
        except Exception:
            pass
        if _log_file is not None:
            try:
                _log_file.write(line + "\n")
                _log_file.flush()
            except Exception:
                pass


def log_exc(where: str):
    log(f"ОШИБКА в {where}: {traceback.format_exc(limit=6)}")


class Budget:
    """Бюджет времени на задачу. Все этапы спрашивают у него, сколько осталось."""

    def __init__(self, total_seconds: float):
        self.total = float(total_seconds)

    def left(self) -> float:
        return self.total - elapsed()

    def frac_used(self) -> float:
        return elapsed() / self.total

    def has(self, seconds: float) -> bool:
        return self.left() > seconds


def effective_cpus(cap: int = 8) -> int:
    """Число доступных ядер с учётом affinity и квоты cgroup (os.cpu_count видит весь хост)."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    quota = None
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts and parts[0] != "max":
            quota = float(parts[0]) / float(parts[1])
    except Exception:
        pass
    if quota is None:
        try:
            q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                quota = q / p
        except Exception:
            pass
    if quota is not None:
        n = min(n, max(1, int(quota + 0.5)))
    return max(1, min(n, cap))


def total_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        return 32.0


def atomic_write_text(path: Path, text: str):
    """Запись через временный файл в том же каталоге (os.replace не работает между файловыми системами)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_json(path: Path, obj):
    try:
        atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    except Exception:
        log_exc("save_json")


def short(obj, limit: int = 600) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def python_exe() -> str:
    return sys.executable or "python3"


def pick_threads(n_max: int) -> int:
    """Замер LightGBM на малых данных с n, n/2 и 1 потоком: при перегруженных ядрах (соседний процесс)
    полное число потоков может работать в разы медленнее — берём самое быстрое значение."""
    if n_max <= 1:
        return 1
    try:
        import time as _t

        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20000, 20)).astype("float32")
        y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
        ds = lgb.Dataset(X, y, params={"verbose": -1}).construct()
        timings = {}
        for n in sorted({n_max, max(1, n_max // 2), 1}, reverse=True):
            t0 = _t.time()
            lgb.train({"objective": "binary", "num_leaves": 31, "verbose": -1, "num_threads": n}, ds, 40)
            timings[n] = _t.time() - t0
            if n == n_max and timings[n] < 0.5:
                break  # всё быстро — ядра свободны
        fastest = min(timings.values())
        # наибольшее число потоков, которое не хуже самого быстрого варианта более чем на 25%
        best = max(n for n, t in timings.items() if t <= 1.25 * fastest)
        log("замер потоков:", {k: round(v, 2) for k, v in timings.items()}, "-> берём", best)
        return best
    except Exception:
        return n_max
