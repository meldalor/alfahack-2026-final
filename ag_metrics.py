"""Метрики, их направление и правила принятия решения по вероятностям."""

import numpy as np
from sklearn import metrics as skm

# имя -> (направление, тип задачи, нужен ли вероятностный выход)
METRICS = {
    "rmse": ("min", "regression", False),
    "mse": ("min", "regression", False),
    "mae": ("min", "regression", False),
    "rmsle": ("min", "regression", False),
    "mape": ("min", "regression", False),
    "smape": ("min", "regression", False),
    "r2": ("max", "regression", False),
    "medae": ("min", "regression", False),
    "f1_macro": ("max", "classification", False),
    "f1_weighted": ("max", "classification", False),
    "f1_micro": ("max", "classification", False),
    "f1": ("max", "classification", False),
    "accuracy": ("max", "classification", False),
    "balanced_accuracy": ("max", "classification", False),
    "mcc": ("max", "classification", False),
    "qwk": ("max", "classification", False),
    "roc_auc": ("max", "classification", True),
    "average_precision": ("max", "classification", True),
    "logloss": ("min", "classification", True),
}

PROBA_METRICS = {m for m, v in METRICS.items() if v[2]}
# метрики по меткам, для которых веса классов/порог подбираются на OOF
TUNABLE = {"f1_macro", "f1_weighted", "balanced_accuracy", "f1", "mcc", "qwk", "accuracy", "f1_micro"}


def direction(metric: str) -> str:
    return METRICS.get(metric, ("max",))[0]


def better(a: float, b: float, metric: str) -> bool:
    """a лучше b?"""
    if b is None or not np.isfinite(b):
        return a is not None and np.isfinite(a)
    if a is None or not np.isfinite(a):
        return False
    return a < b if direction(metric) == "min" else a > b


def gain(new: float, old: float, metric: str) -> float:
    """Положительное число = улучшение."""
    return (old - new) if direction(metric) == "min" else (new - old)


def regression_score(metric: str, y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    if metric == "mse":
        return float(np.mean((y - p) ** 2))
    if metric == "mae":
        return float(np.mean(np.abs(y - p)))
    if metric == "medae":
        return float(np.median(np.abs(y - p)))
    if metric == "rmsle":
        return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))
    if metric == "mape":
        return float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-9)))
    if metric == "smape":
        return float(np.mean(2 * np.abs(y - p) / np.maximum(np.abs(y) + np.abs(p), 1e-9)))
    if metric == "r2":
        return float(skm.r2_score(y, p))
    return float(np.sqrt(np.mean((y - p) ** 2)))


def decide(proba: np.ndarray, metric: str, params: dict | None = None) -> np.ndarray:
    """Вероятности (n, K) -> индексы классов с учётом подобранных весов или порога."""
    params = params or {}
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] == 2 and "threshold" in params:
        return (proba[:, 1] >= params["threshold"]).astype(int)
    w = params.get("weights")
    if w is not None:
        return np.argmax(proba * np.asarray(w)[None, :], axis=1)
    return np.argmax(proba, axis=1)


FAST_LABEL_METRICS = {"f1_macro", "f1_weighted", "f1_micro", "accuracy", "balanced_accuracy"}


def label_score(metric: str, y, labels, n_classes: int, sample_weight=None) -> float:
    """Метрика по меткам; sample_weight — веса строк (например, оценка сдвига частот классов в тесте)."""
    if metric in FAST_LABEL_METRICS or (metric == "f1" and n_classes == 2):
        # через матрицу ошибок (bincount): в десятки раз быстрее sklearn, значения те же
        y = np.asarray(y, dtype=np.int64)
        lab = np.asarray(labels, dtype=np.int64)
        cm = np.bincount(y * n_classes + lab, weights=sample_weight,
                         minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(float)
        tp, support, predicted, total = np.diag(cm), cm.sum(1), cm.sum(0), cm.sum()
        if total <= 0:
            return 0.0
        if metric in ("accuracy", "f1_micro"):
            return float(tp.sum() / total)
        if metric == "balanced_accuracy":
            present = support > 0
            return float(np.mean(tp[present] / support[present]))
        denom = support + predicted
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1.0), 0.0)
        if metric == "f1":
            return float(f1[1])
        if metric == "f1_weighted":
            return float(np.average(f1, weights=support)) if support.sum() > 0 else 0.0
        return float(f1.mean())
    sw = sample_weight
    if metric == "f1":
        return float(skm.f1_score(y, labels, average="macro", zero_division=0, sample_weight=sw))
    if metric == "mcc":
        return float(skm.matthews_corrcoef(y, labels, sample_weight=sw))
    if metric == "qwk":
        return float(skm.cohen_kappa_score(y, labels, weights="quadratic", sample_weight=sw))
    return float(skm.accuracy_score(y, labels, sample_weight=sw))


def proba_score(metric: str, y, proba: np.ndarray) -> float:
    n_classes = proba.shape[1]
    if metric == "roc_auc":
        if n_classes == 2:
            return float(skm.roc_auc_score(y, proba[:, 1]))
        p = np.clip(proba, 1e-12, None)
        p = p / p.sum(1, keepdims=True)
        return float(skm.roc_auc_score(y, p, multi_class="ovr", labels=np.arange(n_classes)))
    if metric == "average_precision":
        if n_classes == 2:
            return float(skm.average_precision_score(y, proba[:, 1]))
        onehot = np.eye(n_classes)[y]
        return float(skm.average_precision_score(onehot, proba, average="macro"))
    if metric == "logloss":
        p = np.clip(proba, 1e-15, 1)
        p = p / p.sum(1, keepdims=True)
        return float(skm.log_loss(y, p, labels=np.arange(n_classes)))
    raise ValueError(metric)


def score(metric: str, task_type: str, y, pred, decision: dict | None = None, sample_weight=None) -> float:
    """Единая оценка: pred — значения (регрессия) или вероятности (n, K) (классификация).
    sample_weight учитывается для метрик по меткам."""
    if task_type == "regression":
        return regression_score(metric, y, pred)
    pred = np.asarray(pred, dtype=float)
    if metric in PROBA_METRICS:
        return proba_score(metric, y, pred)
    return label_score(metric, y, decide(pred, metric, decision), pred.shape[1], sample_weight)


def optimize_decision(metric: str, y, proba: np.ndarray, rounds: int = 2, sample_weight=None) -> dict:
    """Подбор весов классов (мультикласс) или порога (бинарная) под метрику по меткам."""
    proba = np.asarray(proba, dtype=float)
    n_classes = proba.shape[1]
    if metric in PROBA_METRICS or metric not in TUNABLE:
        return {}
    sw = sample_weight
    base = label_score(metric, y, decide(proba, metric), n_classes, sw)
    if n_classes == 2:
        best_t, best = 0.5, base
        for t in np.unique(np.quantile(proba[:, 1], np.linspace(0.01, 0.99, 197))):
            s = label_score(metric, y, (proba[:, 1] >= t).astype(int), 2, sw)
            if s > best + 1e-12:
                best, best_t = s, float(t)
        return {"threshold": best_t, "score": best, "base": base}
    w = np.ones(n_classes)
    best = base
    grid = np.exp(np.linspace(-2.5, 2.5, 21 if n_classes <= 30 else 9))
    if n_classes > 30:
        rounds = 1
    for _ in range(rounds):
        improved = False
        for k in range(n_classes):
            keep = w[k]
            for g in grid:
                w[k] = g
                s = label_score(metric, y, np.argmax(proba * w[None, :], axis=1), n_classes, sw)
                if s > best + 1e-9:
                    best, keep, improved = s, g, True
            w[k] = keep
        if not improved:
            break
    return {"weights": w.tolist(), "score": best, "base": base}


def fold_scores(metric, task_type, y, pred, folds, decision=None, sample_weight=None):
    out = []
    for _, va in folds:
        try:
            out.append(score(metric, task_type, y[va], pred[va], decision,
                             None if sample_weight is None else sample_weight[va]))
        except Exception:
            out.append(float("nan"))
    return out
