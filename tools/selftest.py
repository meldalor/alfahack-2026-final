"""Самопроверка агента на синтетических задачах разных типов (без настоящих данных кейса).

python tools/selftest.py <рабочий каталог> [секунд на задачу]

Создаёт задачи: бинарная с ROC-AUC и вероятностями, регрессия с MAE, датой и строковым id,
мультикласс с числовыми метками и CSV через «;», классификация текстов; прогоняет агента
как на платформе и проверяет формат submission.csv.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def make_tasks(base: Path):
    rng = np.random.default_rng(0)
    tasks = {}

    # бинарная: вероятность положительного класса, метрика ROC-AUC
    n = 6000
    X = rng.normal(size=(n, 6))
    p = 1 / (1 + np.exp(-(X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3])))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["segment"] = rng.choice(["a", "b", "c"], n)
    df["churn"] = np.where(rng.random(n) < p, "yes", "no")
    df.insert(0, "client_id", np.arange(n) + 10_000)
    tasks["binary"] = (df, "client_id", "churn", 0.5,
                       "# Отток\n\nБинарная классификация: уйдёт ли клиент (`churn`: yes/no).\n\n"
                       "Данные: `binary/participants/train.csv`, `binary/participants/test.csv`.\n\n"
                       "Метрика — **ROC-AUC**, в ответе вероятность `yes`.\n")

    # регрессия: MAE, дата, строковый id
    n = 5000
    d = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    x = rng.normal(size=(n, 4))
    y = 50 + 10 * x[:, 0] + 5 * np.sin(d.dayofyear / 58) + rng.normal(0, 3, n)
    df = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    df["date"] = d.strftime("%Y-%m-%d")
    df["sales"] = y
    df.insert(0, "row", [f"r{i}" for i in range(n)])
    tasks["regression"] = (df, "row", "sales", 0.0,
                           "# Продажи\n\nРегрессия `sales`.\n\nФайлы: `train.csv`, `test.csv`, `sample_submission.csv`.\n\n"
                           "Метрика оценки — MAE.\n")

    # мультикласс: числовые метки, разделитель «;»
    n = 4000
    X = rng.normal(size=(n, 8))
    cls = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0.5).astype(int) + 1
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    df["grade"] = cls
    df.insert(0, "ID", np.arange(n))
    tasks["multiclass"] = (df, "ID", "grade", 1,
                           "# Классы\n\nМногоклассовая классификация `grade`.\n\nОсновная метрика — **Macro F1**.\n")

    # тексты
    n = 3000
    words = {"neg": "bad awful broken slow worst".split(), "pos": "great love perfect fast best".split()}
    common = "the a product delivery item service price box order shop".split()
    lab = rng.choice(["neg", "pos"], n)
    text = [" ".join(list(rng.choice(common, 8)) + [rng.choice(words[c])]) for c in lab]
    df = pd.DataFrame({"id": np.arange(n), "text": text, "label": lab})
    tasks["text"] = (df, "id", "label", "pos",
                     "# Отзывы\n\nКлассификация текста `text` по тональности `label`.\n\nМетрика: Accuracy.\n")

    # Text with a strictly chronological train/test split.
    temporal = df.copy()
    temporal["timestamp"] = pd.date_range("2024-01-01", periods=len(temporal), freq="h").astype(str)
    tasks["temporal_text"] = (temporal, "id", "label", "pos",
        "# Reviews over time\nPredict label from text. Metric: Accuracy. Validate chronologically.\n")

    # два таргета в одном шаблоне: регрессия и бинарная метка
    n = 3000
    X = rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(5)])
    df["amount"] = 3 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.5, n)
    df["is_big"] = (X[:, 2] + X[:, 3] > 0).astype(int)
    df.insert(0, "uid", np.arange(n))
    tasks["multitarget"] = (df, "uid", ["amount", "is_big"], 0,
                            "# Два ответа\n\nПредсказать `amount` (регрессия, RMSE) и `is_big` (0/1).\n\n"
                            "Метрика: среднее RMSE и accuracy.\n")

    out = {}
    for name, (df, id_col, target, fill, md) in tasks.items():
        d = base / name
        if d.exists():
            shutil.rmtree(d)
        nested = d / name / "participants" if "participants" in md else d
        nested.mkdir(parents=True, exist_ok=True)
        cut = int(len(df) * 0.75)
        sep = ";" if name == "multiclass" else ","
        targets = target if isinstance(target, list) else [target]
        df.iloc[:cut].to_csv(nested / "train.csv", index=False, sep=sep)
        df.iloc[cut:].drop(columns=targets).to_csv(nested / "test.csv", index=False, sep=sep)
        pd.DataFrame({id_col: df[id_col].iloc[cut:], **{t: fill for t in targets}}).to_csv(
            nested / "sample_submission.csv", index=False)
        (d / "task.md").write_text(md, encoding="utf-8")
        out[name] = (d, df.iloc[cut:], id_col, target)
    return out


def main():
    work = Path(sys.argv[1]).resolve()
    seconds = sys.argv[2] if len(sys.argv) > 2 else "180"
    tasks = make_tasks(work / "tasks")
    only = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--only=")), None)
    if only:
        tasks = {name: item for name, item in tasks.items() if name in only.split(",")}
        if not tasks:
            raise SystemExit("No matching selftest tasks")
    env = dict(os.environ, AGENT_TIME_BUDGET=seconds, AGENT_HARD_LIMIT=str(int(seconds) + 120))
    env.pop("OPENAI_API_KEY", None)
    env.pop("PGDATABASE", None)
    ok = True
    for name, (d, test, id_col, target) in tasks.items():
        for f in ROOT.glob("*.py"):
            shutil.copy(f, d / f.name)
        shutil.copy(ROOT / "requirements.txt", d / "requirements.txt")
        subprocess.call([sys.executable, "solution.py", "task.md"], cwd=d, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sub = d / "submission.csv"
        problems = []
        if not sub.exists():
            problems.append("нет submission.csv")
        else:
            s = pd.read_csv(sub)
            if list(s.columns) != [id_col] + (target if isinstance(target, list) else [target]):
                problems.append(f"колонки {list(s.columns)}")
            if len(s) != len(test) or s.isna().any().any():
                problems.append("число строк или пропуски")
            elif not (s[id_col].astype(str).values == test[id_col].astype(str).values).all():
                problems.append("порядок id")
        ok &= not problems
        print(f"{name:12s} {'OK' if not problems else 'ПРОБЛЕМЫ: ' + '; '.join(problems)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
