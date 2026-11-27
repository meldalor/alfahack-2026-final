"""Точка входа агента: python3 solution.py task.md

Агент читает условие, находит данные, определяет тип задачи и метрику, проводит цикл экспериментов
(AutoResearch) и сохраняет submission.csv рядом с task.md. Первый корректный файл ответа появляется
в первые минуты и дальше только улучшается, поэтому остановка по времени не оставляет задачу без ответа.
"""

import dataclasses
import os
import re
import sys
from pathlib import Path

from ag_utils import effective_cpus

N_JOBS = effective_cpus()
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(N_JOBS))

import threading  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ag_ensemble import (LabelShift, apply_zero_mape_hurdle, blend, choose_shrink_power, decide_on_test,  # noqa: E402
                         finalize_classification, rank_blend, stack_blend)
from ag_features import (causal_stream_features, cross_impute_outages, detect_time_split, entity_periods,  # noqa: E402
                         op_entity_history, prepare)
from ag_llm import LLM  # noqa: E402
from ag_metrics import PROBA_METRICS, RANK_METRICS, TUNABLE, better, decide, direction, oof_score, score  # noqa: E402
from ag_models import (FitResult, ModelConfig, adversarial_scores, fit_cat, fit_et, fit_hgb, fit_lgb, fit_linear,  # noqa: E402
                       fit_mlp, fit_xgb, make_plan, safe_fit)
from ag_research import Researcher  # noqa: E402
from ag_task import build_spec, find_files, llm_refine_spec, read_table  # noqa: E402
from ag_utils import Budget, atomic_write_text, elapsed, log, log_exc, pick_threads, save_json, setup_log, short  # noqa: E402

TIME_BUDGET = float(os.environ.get("AGENT_TIME_BUDGET", 1.80 * 3600))  # плановое время работы
HARD_LIMIT = float(os.environ.get("AGENT_HARD_LIMIT", 1.95 * 3600))  # аварийная остановка до лимита платформы


class SubmissionWriter:
    """Формирует submission.csv в формате шаблона. Запись атомарная: файл всегда целый.

    Хранит текущие значения всех колонок ответа: при нескольких таргетах каждый обновляет только свою.
    """

    def __init__(self, path: Path, spec, test: pd.DataFrame, sample: pd.DataFrame | None = None):
        self.path, self.spec = path, spec
        self.n = len(test)
        self.cols = list(spec.sub_columns)
        cols = self.cols
        targets = {spec.target, *spec.extra_targets, *[str(c) for c in spec.classes]}
        id_candidates = [c for c in cols if c not in targets]
        self.id_name = spec.id_col if spec.id_col in cols else (id_candidates[0] if id_candidates and len(cols) > 1
                                                                 else None)
        if self.id_name is not None and spec.id_col and spec.id_col in test.columns and self.id_name == spec.id_col:
            self.ids = test[spec.id_col].values
        elif self.id_name is not None and self.id_name in test.columns:
            self.ids = test[self.id_name].values
        elif self.id_name is not None and sample is not None and len(sample) == self.n and self.id_name in sample.columns:
            self.ids = sample[self.id_name].values  # id есть только в шаблоне: порядок строк совпадает с test
        elif self.id_name is not None:
            self.ids = np.arange(self.n)
        else:
            self.ids = None
        self.output_order = np.arange(self.n)
        if sample is not None and self.id_name in sample and self.ids is not None:
            source = pd.Index(self.ids)
            desired = pd.Index(sample[self.id_name])
            if not source.is_unique or not desired.is_unique or len(desired) != self.n:
                raise ValueError("submission: IDs must be unique and template length must match")
            order = source.get_indexer(desired)
            if (order < 0).any():
                order = source.astype(str).get_indexer(desired.astype(str))
            if (order < 0).any():
                raise ValueError("submission: template IDs differ from test IDs")
            self.output_order = order
            self.ids = sample[self.id_name].values[np.argsort(order)]
        self.values = {}
        self.versions = 0

    def out_col(self, spec) -> str:
        """Колонка ответа: таргет, если он есть в шаблоне, иначе первая не-id колонка шаблона."""
        others = [c for c in self.cols if c != self.id_name]
        return spec.target if spec.target in self.cols or not others else others[0]

    def set_values(self, test_pred: np.ndarray, decision: dict | None = None, train_y=None, spec=None):
        spec = spec or self.spec
        if spec.task_type == "regression":
            p = np.asarray(test_pred, dtype=float).reshape(-1)
            fill = float(np.nanmedian(p)) if np.isfinite(p).any() else 0.0
            p = np.where(np.isfinite(p), p, fill)
            if train_y is not None and len(train_y):
                p = np.clip(p, float(np.min(train_y)), float(np.max(train_y)))
            values = {self.out_col(spec): p}
        else:
            proba = np.asarray(test_pred, dtype=float)
            proba = np.where(np.isfinite(proba), proba, 1.0 / proba.shape[1])
            if spec.output_kind == "proba":
                values = {self.out_col(spec): proba[:, 1]}
            elif spec.output_kind == "proba_multi":
                values = {}
                for i, c in enumerate(spec.classes):
                    cs = (str(int(round(float(c))))
                          if isinstance(c, (float, np.floating)) and np.isfinite(c) and float(c).is_integer()
                          else str(c).strip().lower())
                    for col in self.cols:
                        if col == self.id_name or col in values:
                            continue
                        s = str(col).strip().lower()
                        if s == cs or any(s.startswith(pfx) and s[len(pfx):] == cs
                                          for pfx in ("class_", "prob_", "proba_", "pred_", "p_", "target_", "label_")):
                            values[col] = proba[:, i]
            else:
                idx = decide(proba, spec.metric, decision)
                labels = np.asarray(spec.classes, dtype=object)[idx]
                if all(isinstance(c, (float, np.floating)) and float(c).is_integer() for c in spec.classes):
                    labels = labels.astype(float).astype(np.int64)  # 1.0 -> 1, как в исходных метках
                values = {self.out_col(spec): labels}
        if any(len(v) != self.n for v in values.values()):
            raise ValueError("submission: неверная длина прогноза")
        self.values.update(values)

    def flush(self, note: str = ""):
        df = pd.DataFrame(index=range(self.n))
        for c in self.cols:
            if c in self.values:
                df[c] = self.values[c]
            elif self.ids is not None and c == self.id_name:
                df[c] = self.ids
            else:
                raise ValueError(f"submission: нет значений для колонки {c}")
        if df.isna().any().any():
            raise ValueError("submission: пропуски")
        atomic_write_text(self.path, df.iloc[self.output_order].to_csv(index=False))
        self.versions += 1
        log(f"submission.csv записан (версия {self.versions}{', ' + note if note else ''})")

    def write(self, test_pred: np.ndarray, decision: dict | None = None, train_y=None, note: str = "", spec=None):
        self.set_values(test_pred, decision, train_y, spec)
        self.flush(note)

    def set_constant(self, train_y_raw: pd.Series, spec=None):
        spec = spec or self.spec
        if spec.task_type == "regression":
            v = float(pd.to_numeric(train_y_raw, errors="coerce").median())
            self.set_values(np.full(self.n, v), spec=spec)
            return
        counts = train_y_raw.value_counts()
        prior = np.array([counts.get(c, 0) for c in spec.classes], dtype=float)
        self.set_values(np.tile(prior / max(prior.sum(), 1), (self.n, 1)), spec=spec)

    def for_spec(self, spec):
        return _TargetWriter(self, spec)


class _TargetWriter:
    """Запись ответа по одному таргету (остальные колонки берутся из уже записанных значений)."""

    def __init__(self, writer: SubmissionWriter, spec):
        self.writer, self.spec = writer, spec

    def write(self, test_pred, decision=None, train_y=None, note: str = ""):
        prefix = f"[{self.spec.target}] " if self.writer.spec.extra_targets else ""
        self.writer.write(test_pred, decision, train_y, prefix + note, spec=self.spec)


def start_watchdog(writer_ref: dict, out_path: Path):
    def fire():
        log("WATCHDOG: аварийное завершение по времени")
        try:
            if not out_path.exists() and writer_ref.get("fallback"):
                writer_ref["fallback"]()
        except Exception:
            log_exc("watchdog fallback")
        os._exit(0)

    delay = max(10.0, HARD_LIMIT - elapsed())
    t = threading.Timer(delay, fire)
    t.daemon = True
    t.start()
    return t


def _resolve_db_tables(tables: dict, text: str) -> dict:
    """Resolve train/test/sample/dictionary table names flexibly (case-insensitive & patterned)."""
    from ag_task import _role_of

    out = {}
    by_low = {k.lower(): k for k in tables}
    for role, exacts in (
        ("train", ("train", "training")),
        ("test", ("test", "testing")),
        ("sample", ("sample_submission", "submission_sample", "sample_sub")),
        ("dictionary", ("data_dictionary", "dictionary", "metadata")),
    ):
        for ex in exacts:
            if ex in by_low:
                out[role] = by_low[ex]
                break
    # Fallback: match by role pattern (e.g., events_train / train_data / test_events)
    for role in ("train", "test", "sample", "dictionary"):
        if role in out:
            continue
        cands = [k for k in tables if k not in out.values() and _role_of(k) == role]
        if cands:
            low_txt = text.lower()
            cands.sort(key=lambda k: (k.lower() not in low_txt, len(k), k))
            out[role] = cands[0]
    return out


def run(task_path: Path):
    base = task_path.parent
    work = base / "agent_work"
    try:
        work.mkdir(parents=True, exist_ok=True)
    except Exception:
        import tempfile

        work = Path(tempfile.mkdtemp(prefix="agent_work_"))  # каталог задачи только для чтения
    setup_log(work / "agent.log")
    global N_JOBS
    N_JOBS = pick_threads(N_JOBS)
    out_path = base / "submission.csv"
    writer_ref = {}
    start_watchdog(writer_ref, out_path)
    log(f"старт: задача {task_path}, ядер {N_JOBS}, бюджет {TIME_BUDGET:.0f}s")

    # 1. условие и данные
    text = task_path.read_text(encoding="utf-8", errors="replace")
    files = find_files(task_path, text, search_parent=not bool(os.environ.get("PGDATABASE")))
    db_notes = []
    rel_meta = {}
    if os.environ.get("PGDATABASE") or "train" not in files or "test" not in files:
        from ag_database import extract
        from ag_relational import enrich
        try:
            tables = extract(work, meta=rel_meta)
        except TypeError:
            tables = extract(work)
        role_map = _resolve_db_tables(tables, text)
        if "train" not in role_map or "test" not in role_map:
            raise RuntimeError(f"Database must expose train and test tables (found: {sorted(tables)})")
        log("таблицы БД:", role_map)
        db_train, db_test = enrich(tables[role_map["train"]], tables[role_map["test"]], tables,
                                   skip_names=role_map.values(), report=db_notes, meta=rel_meta)
        files = {k: v for k, v in files.items() if k in ("sample", "dictionary")}
        dic_df = tables.get(role_map.get("dictionary"))
        if dic_df is None and rel_meta.get("column_comments"):
            try:
                from ag_database import comments_to_dictionary
                dic_df = comments_to_dictionary(
                    rel_meta["column_comments"],
                    main_tables=[role_map["train"], role_map["test"]],
                )
            except Exception:
                log_exc("comments_to_dictionary")
        for role, frame in (("train", db_train), ("test", db_test),
                            ("sample", tables.get(role_map.get("sample"))),
                            ("dictionary", dic_df)):
            if frame is not None:
                path = work / f"db_{role}.parquet"
                frame.to_parquet(path, index=False)
                files[role] = str(path)
    log("файлы:", files)
    if "train" not in files or "test" not in files:
        raise RuntimeError(f"не найдены train/test: {files}")
    train = read_table(Path(files["train"]))
    test = read_table(Path(files["test"]))
    sample = read_table(Path(files["sample"])) if "sample" in files else None
    dic = None
    if "dictionary" in files:
        try:
            dic = read_table(Path(files["dictionary"]))
        except Exception:
            log_exc("read dictionary")
    log(f"train {train.shape}, test {test.shape}, sample {None if sample is None else sample.shape}")

    spec = build_spec(task_path, train, test, sample, dic, files)
    if db_notes:
        spec.notes = list(db_notes[:12]) + list(spec.notes)
    log("спецификация:", short(spec.public(), 1200))
    try:
        # первый страховочный ответ ещё до обращения к LLM
        early = SubmissionWriter(out_path, spec, test, sample)
        for t in [spec.target] + list(spec.extra_targets):
            sp0 = spec if t == spec.target else build_spec(task_path, train, test, sample, dic, files, force_target=t)
            early.set_constant(train[t].dropna(), sp0)
        early.flush("запасной")
        writer_ref["fallback"] = lambda: early.flush("запасной")
    except Exception:
        log_exc("early fallback")
    llm = LLM(time_limit=0.3 * TIME_BUDGET)
    hints = {}
    try:
        hints = llm_refine_spec(spec, llm, train, test, sample)
    except Exception:
        log_exc("llm_refine_spec")
    specs = [spec]
    for t in spec.extra_targets:
        try:
            specs.append(build_spec(task_path, train, test, sample, dic, files, force_target=t))
        except Exception:
            log_exc(f"spec для {t}")
    if len(specs) > 1:
        log(f"несколько таргетов: {[s.target for s in specs]} — решаем по очереди")

    # страховочный ответ по всем таргетам сразу
    writer = SubmissionWriter(out_path, spec, test, sample)
    for sp in specs:
        writer.set_constant(train[sp.target].dropna(), sp)
    writer_ref["fallback"] = lambda: writer.flush("запасной")
    writer.flush("запасной: медиана / априорные частоты")
    save_json(work / "spec.json", {**spec.public(), "llm": hints})

    for i, sp in enumerate(specs):
        window = max(60.0, (TIME_BUDGET - elapsed()) / (len(specs) - i))
        budget = Budget(elapsed() + window)
        twork = work if len(specs) == 1 else work / f"target_{i + 1}"
        twork.mkdir(parents=True, exist_ok=True)
        try:
            solve_target(sp, train, test, llm, hints if i == 0 else {}, budget, writer.for_spec(sp), twork,
                         rel_meta=rel_meta)
        except Exception:
            log_exc(f"solve_target {sp.target}")
    save_json(work / "llm_stats.json", llm.stats())
    log("готово")


def solve_target(spec, train, test, llm, hints, budget, writer, work, rel_meta=None):
    global N_JOBS
    start = elapsed()
    log(f"задача: {spec.task_type}, метрика {spec.metric} ({spec.metric_source}), таргет {spec.target}, "
        f"id {spec.id_col}, вывод {spec.output_kind}, время до {budget.total:.0f}s")

    # 2. признаки и схема валидации
    tr_in, te_in = train, test
    try:
        st_tr, st_te = causal_stream_features(train, test, exclude={spec.target, *spec.extra_targets, spec.id_col})
        if st_tr.shape[1] > 0:
            tr_in = pd.concat([train.reset_index(drop=True), st_tr], axis=1)
            te_in = pd.concat([test.reset_index(drop=True), st_te], axis=1)
    except Exception:
        log_exc("causal_stream_features")
    prep = prepare(tr_in, te_in, spec)
    prep.X, prep.Xt = cross_impute_outages(prep.X, prep.Xt, prep.cat_cols)
    y = prep.y
    log(f"признаков {prep.X.shape[1]}, категориальных {len(prep.cat_cols)}, текстовых {len(prep.text_cols)}")
    time_col = detect_time_split(prep.X, prep.Xt, prep.cat_cols)
    if time_col:
        log(f"обнаружен временной сплит по {time_col}: тест целиком позже обучения")
    plan = make_plan(prep.X, y, spec.task_type, time_col=time_col)
    log("валидация:", plan.desc)

    # 2б. строго каузальный expanding-window 1-step-ahead nowcaster для регулярных панелей
    if rel_meta and rel_meta.get("panels") and plan.time_col:
        try:
            from ag_nowcast import build_nowcast_features

            valid_y = prep.info.get("valid_y_mask")
            if valid_y is None:
                valid_y = train[spec.target].notna().to_numpy()
            rm = dict(rel_meta)
            if rm.get("train_times") is not None and len(rm["train_times"]) == len(valid_y):
                rm["train_times"] = rm["train_times"].loc[valid_y].reset_index(drop=True)
            if rm.get("train_groups") is not None and len(rm["train_groups"]) == len(valid_y):
                rm["train_groups"] = rm["train_groups"].loc[valid_y].reset_index(drop=True)
            nc_lim = min(180.0, max(30.0, 0.15 * budget.left()))
            nc_tr, nc_te = build_nowcast_features(rm, y, plan, spec.task_type, n_jobs=N_JOBS, max_seconds=nc_lim)
            if nc_tr is not None and nc_te is not None and len(nc_tr) == len(prep.X):
                existing = set(prep.X.columns)
                ren, keep = {}, []
                for c in nc_tr.columns:
                    safe = re.sub(r"[^\w]+", "_", str(c)).strip("_") or "nc"
                    while safe in existing or safe in ren.values():
                        safe += "_"
                    ren[c] = safe
                    if nc_tr[c].nunique(dropna=True) > 1:
                        keep.append(c)
                if keep:
                    a_nc = nc_tr[keep].rename(columns=ren).reset_index(drop=True)
                    b_nc = nc_te[keep].rename(columns=ren).reset_index(drop=True)
                    prep.X = pd.concat([prep.X.reset_index(drop=True), a_nc], axis=1)
                    prep.Xt = pd.concat([prep.Xt.reset_index(drop=True), b_nc], axis=1)
                    log(f"nowcast: добавлено {len(keep)} признаков -> всего {prep.X.shape[1]}")
        except Exception:
            log_exc("build_nowcast_features")
    raw_hints = hints.get("raw", {}) if isinstance(hints.get("raw"), dict) else {}
    shift = {"adv": None, "auc": None, "top": [], "label": None}
    try:
        shift["adv"], shift["auc"], shift["top"] = adversarial_scores(prep.X, prep.Xt, N_JOBS)
        log(f"adversarial validation: AUC train/test = {shift['auc']:.3f}, сильнее всего различаются {shift['top']}")
    except Exception:
        log_exc("adversarial_scores")

    # 3. AutoResearch
    researcher = Researcher(spec, prep, plan, llm, budget, N_JOBS, work, llm_hints=raw_hints)
    if shift["auc"] is not None:
        researcher.shift_note = (f"train/test covariate shift: adversarial AUC={shift['auc']:.3f} "
                                 f"(0.5 = identical); most shifted features: {shift['top']}")
    try:
        res_frac = float(os.environ.get("AGENT_RESEARCH_FRAC", "0.30"))
    except Exception:
        res_frac = 0.30
    res_frac = min(0.60, max(0.10, res_frac))
    research_deadline = start + res_frac * (budget.total - start)
    log(f"бюджет AutoResearch: {int(round(res_frac * 100))}% (до {research_deadline:.0f}s), "
        f"финальные модели: {int(round((1.0 - res_frac) * 100))}%")

    def on_baseline(r):
        ey, eplan = researcher.ey, researcher.eplan
        m = eplan.oof_mask & ~np.isnan(r.oof).reshape(len(ey), -1).any(1)
        if spec.task_type == "regression":
            writer.write(r.test, train_y=y, note="базовая LightGBM")
            shift["base_written"] = True
        else:
            adv = shift["adv"][researcher.sub] if (shift["adv"] is not None and researcher.sub is not None) else shift["adv"]
            ls = None
            if researcher.sub is None and not r.config.get("time_cut"):
                # сдвиг долей классов оценивается по базовой модели: она обучена на всех строках и не обрезана
                # по времени (у финальной модели фолд может не доучиться, и оценка по ней уходит в сторону)
                ls = shift["label"] = label_shift(spec, prep, ey, m, r.oof, r.test, shift["auc"], plan.time_col)
                researcher.label_shift = ls
            power, _ = choose_shrink_power(spec, ey, m, r.oof, adv, shift["auc"], shift=ls)
            decision, test_adj, _ = finalize_classification(spec, ey, m, r.oof, r.test, 0.0, weight_power=power,
                                                            shift=ls)
            writer.write(test_adj, decision=decision, note="базовая LightGBM")
            shift["base_written"] = True

    try:
        researcher.run(deadline=research_deadline, on_baseline=on_baseline)
    except Exception:
        log_exc("research")
    X, Xt = researcher.frames()
    cfg = researcher.cfg
    log(f"исследование завершено: лучший {spec.metric}={researcher.best_score}, конфигурация {short(cfg.to_dict(), 400)}")

    # 4. финальные модели, смесь и ответ (число потоков перепроверяем: соседний процесс мог появиться позже)
    N_JOBS = pick_threads(effective_cpus())
    final_stage(spec, prep, plan, X, Xt, y, cfg, researcher, writer, budget, work, train, shift)
    write_report(work, spec, plan, researcher, llm)


def extra_entity_keys(researcher, n: int):
    """Ключи объекта, история по которым в исследовании проиграла не больше 2% (лучшая из попыток), — до n штук."""
    if n <= 0 or researcher.best_score is None:
        return []
    accepted, best = set(), {}
    for e in researcher.journal:
        if e.get("kind") != "entity_history" or not isinstance(e.get("args", {}).get("key"), list):
            continue
        k = tuple(e["args"]["key"])
        if e.get("accepted"):
            accepted.add(k)
        elif e.get("delta") is not None:
            best[k] = max(best.get(k, -np.inf), float(e["delta"]))
    lim = -0.02 * abs(researcher.best_score)
    cands = sorted(((d, k) for k, d in best.items() if k not in accepted and d > lim), reverse=True)
    return [list(k) for _, k in cands[:n]]


def model_family(name: str) -> str:
    """Семейство модели: сиды одной конфигурации (lgb_main и lgb_s3, xgb и xgb_s2, ...) усредняются вместе."""
    if name == "lgb_main" or re.fullmatch(r"lgb_s\d+", name):
        return "lgb"
    if name == "lgb_huber" or re.fullmatch(r"lgb_huber_s\d+", name):
        return "lgb_huber"
    if name == "lgb_l1" or re.fullmatch(r"lgb_l1_s\d+", name):
        return "lgb_l1"
    if name == "lgb_et" or re.fullmatch(r"lgb_et_s\d+", name):
        return "lgb_et"
    if name.startswith("lgb_offset"):
        return "lgb_offset"
    m = re.fullmatch(r"(lgb_xoffset\d+)_s\d+", name)
    if m:
        return m.group(1)
    for f in ("xgb", "xgb_ph", "cat", "cat_rmse", "cat_top", "cat_d8", "et", "hgb", "mlp_dual", "mlp"):
        if name == f or re.fullmatch(f + r"_s\d+", name):
            return f
    return name


def pool_results(name, rs, y, mask, spec, plan=None):
    """Среднее прогнозов нескольких моделей как одна модель (FitResult)."""
    oof = sum(r.oof for r in rs) / len(rs)
    test = sum(r.test for r in rs) / len(rs)
    s = oof_score(spec.metric, spec.task_type, y, oof, plan=plan, mask=mask)
    return FitResult(name=name, oof=oof, test=test, score=s, fold_scores=[], iters=[], importance=None,
                     seconds=sum(r.seconds for r in rs))


def label_shift(spec, prep, y, mask, oof, test, adv_auc=None, time_col=None):
    """Оценка сдвига долей классов в тесте (для метрик по меткам); None, если неприменимо или не удалось.

    Если train и тест разделяются почти идеально (adversarial AUC ≥ 0.98) не по времени, тест лежит вне области
    train (новые значения категорий, другой диапазон признаков): прогноз там — экстраполяция, и EM принимает его
    смещение за сдвиг долей. На diamonds (огранка, тест — другие цвета) EM оценила долю Ideal в 5% при 37%, и
    macro F1 с поправкой 0.58 против 0.78 без неё; на UNSW-NB15 (AUC 0.89) поправка нужна (+2 к S_cls)."""
    if spec.metric in PROBA_METRICS or spec.metric not in TUNABLE or len(prep.X) != len(y):
        return None
    if adv_auc is not None and adv_auc >= 0.98 and not time_col:
        log(f"сдвиг долей классов не оценивается: train и тест почти не пересекаются (adversarial AUC {adv_auc:.3f})")
        return None
    try:
        return LabelShift(prep.X, prep.Xt, y, mask, oof.shape[1]).fit(oof, test)
    except Exception:
        log_exc("label_shift")
        return None


def test_like_weights(spec, shift, n_train, n_test, time_col=None):
    """Веса строк train для финальных моделей мультикласса, если тест заметно отличается от train: корень из
    отношения плотностей теста и train по adversarial validation, sqrt(adv / (1 − adv) · n_train / n_test).

    Модели учатся (и останавливаются) с упором на строки, похожие на тест. На UNSW-NB15 (обучение на строках train,
    непохожих на тест, оценка на похожих; метки известны) macro F1 с правилом под тест: 0.546 -> 0.560 (80/20),
    0.480 -> 0.535 (50/50), при мягком сдвиге 0.636 -> 0.659 и 0.579 -> 0.590; без сдвига 0.652 -> 0.650. Полные
    веса без корня хуже: их разброс больше. При сплите по времени или почти полном разделении train и теста
    (AUC ≥ 0.98) adversarial-модель отделяет период или область, а не похожие строки, и веса вырождаются — их нет.
    None, если веса не нужны."""
    adv, auc = shift.get("adv"), shift.get("auc")
    if (spec.task_type != "multiclass" or time_col or adv is None or auc is None or not 0.65 <= auc < 0.98
            or len(adv) != n_train):
        return None
    p = np.clip(np.asarray(adv, dtype=float), 1e-6, 1 - 1e-6)
    return np.sqrt(np.clip(p / (1 - p) * n_train / max(n_test, 1), 0.05, 20.0))


def final_stage(spec, prep, plan, X, Xt, y, cfg, researcher, writer, budget, work, train, shift=None):
    results: list[FitResult] = []
    mask = plan.oof_mask
    reserve = min(240.0, max(10.0, 0.08 * budget.left()))
    eval_t = float(np.median(researcher.eval_times)) if researcher.eval_times else 60.0
    state = {"best_val": None, "power": None, "label_shift": None}
    cache = {}
    shift = shift or {}
    row_w = test_like_weights(spec, shift, len(X), len(Xt), plan.time_col)
    if row_w is not None:
        plan = dataclasses.replace(plan, row_weight=row_w)
        # сдвиг долей классов оценивается заново — по первой финальной модели, обученной с этими весами
        shift = {**shift, "label": None}
        log(f"финальные модели учатся с весами похожести на тест: квантили 1/50/99% "
            f"{np.round(np.quantile(row_w, [0.01, 0.5, 0.99]), 3).tolist()}")

    def evaluate_candidate(oof, test):
        m = mask & ~np.isnan(oof).reshape(len(y), -1).any(1)
        if spec.task_type == "regression":
            return score(spec.metric, spec.task_type, y[m], oof[m]), None, test, {}
        if state["power"] is None:
            # степень сжатия весов выбирается один раз — по первой полной модели, чтобы оценки кандидатов были сравнимы
            state["power"], _ = choose_shrink_power(spec, y, m, oof, shift.get("adv"), shift.get("auc"),
                                                    shift=state["label_shift"])
        decision, test_adj, info = finalize_classification(spec, y, m, oof, test, cfg.class_weight_power,
                                                           weight_power=state["power"], shift=state["label_shift"])
        val = decision.get("score")
        if val is None:
            val = oof_score(spec.metric, spec.task_type, y, oof, plan=plan, mask=m)
        return val, decision, test_adj, info

    def publish(note):
        usable = [r for r in results if r is not None and r.test is not None
                  and not np.isnan(r.oof[mask]).reshape(int(mask.sum()), -1).any()]
        if not usable:
            partial = [r for r in results if r is not None and r.test is not None]
            if partial and state["best_val"] is None and not state.get("partial") and not shift.get("base_written"):
                # ни одна модель не прошла все фолды и ответа базовой модели нет: неполная модель всё же лучше константы
                r = partial[0]
                if spec.task_type == "regression":
                    writer.write(r.test, train_y=y, note=f"{note}: неполная модель")
                else:
                    writer.write(r.test, note=f"{note}: неполная модель")
                state["partial"] = True
            return
        if os.environ.get("AGENT_SAVE_MODELS"):
            # для разработки: прогнозы каждой финальной модели, чтобы проверять смеси без переобучения
            mdir = work / "models"
            mdir.mkdir(exist_ok=True)
            if not (mdir / "_meta.npz").exists():
                np.savez_compressed(mdir / "_meta.npz", y=y, mask=mask)
                X.to_pickle(mdir / "X.pkl")
                Xt.to_pickle(mdir / "Xt.pkl")
                save_json(mdir / "cfg.json", cfg.to_dict())
            for r in usable:
                if not (mdir / f"{r.name}.npz").exists():
                    np.savez_compressed(mdir / f"{r.name}.npz", oof=r.oof, test=r.test)
        if spec.task_type != "regression" and not state.get("label_shift_done"):
            # сдвиг долей классов: оценка по базовой модели исследования, иначе — по первой модели без обрезанных
            # по времени фолдов; одна оценка на все кандидаты, чтобы их метрики были сравнимы
            ref = next((r for r in usable if not r.config.get("time_cut")), None)
            if shift.get("label") is not None:
                state["label_shift"], state["label_shift_done"] = shift["label"], True
            elif ref is not None:
                state["label_shift"] = label_shift(spec, prep, y, mask & ~np.isnan(ref.oof).reshape(len(y), -1).any(1),
                                                   ref.oof, ref.test, shift.get("auc"), plan.time_col)
                state["label_shift_done"] = True
        # сиды одной модели сначала усредняются: выбор «удачного» сида по OOF переобучается на валидацию;
        # внутри одного семейства недоученный по времени сид не портит среднее полных сидов, но сильная
        # модель собственного семейства (например, lgb_huber, прошедшая все фолды) никогда не выбрасывается!
        fams = {}
        for r in usable:
            fams.setdefault(model_family(r.name), []).append(r)
        pooled = []
        for f, rs in fams.items():
            uncut = [r for r in rs if not r.config.get("time_cut")]
            if uncut:
                best_uncut = max(uncut, key=lambda r: r.score if direction(spec.metric) == "max" else -r.score).score
                rs = [r for r in rs if (not r.config.get("time_cut")) or better(r.score, best_uncut, spec.metric)]
            pooled.append(rs[0] if len(rs) == 1 else pool_results(f"{f}x{len(rs)}", rs, y, mask, spec, plan=plan))
        scored = []

        def add(name, oof, test, weights):
            single = len(weights) == 1 and name in weights
            try:
                if single and name in cache:
                    res = cache[name]  # одиночная модель уже оценена
                else:
                    res = evaluate_candidate(oof, test)
                    if single:
                        cache[name] = res
            except Exception:
                log_exc("evaluate_candidate")
                return
            scored.append((*res, name, weights, oof, test))

        for r in pooled:
            add(r.name, r.oof, r.test, {r.name: 1.0})
        b = blend(pooled, y, mask, spec, plan=plan)
        if b is not None:
            add("смесь", b["oof"], b["test"], b["weights"])
        rb = rank_blend(pooled, y, mask, spec, plan=plan)
        if rb is not None:
            add("ранг-смесь", rb["oof"], rb["test"], rb["weights"])
        try:
            sb = stack_blend(pooled, y, mask, spec, plan=plan)
            if sb is not None:
                add("стекинг", sb["oof"], sb["test"], sb["weights"])
        except Exception:
            log_exc("stack_blend")
        # средние k лучших семейств: слабое семейство (например, недоученный CatBoost) не тянет вниз всё среднее
        sign = 1.0 if direction(spec.metric) == "max" else -1.0
        order = sorted([sc for sc in scored if sc[4] in {r.name for r in pooled}], key=lambda sc: -sign * sc[0])
        by_name = {r.name: r for r in pooled}
        for k in range(2, len(order) + 1):
            top_r = [by_name[sc[4]] for sc in order[:k]]
            add("среднее" if k == len(pooled) else f"среднее {k} лучших", sum(r.oof for r in top_r) / k,
                sum(r.test for r in top_r) / k, {r.name: round(1 / k, 3) for r in top_r})
        best = None
        for sc in scored:
            if best is None or better(sc[0], best[0], spec.metric):
                best = sc
        if best is None:
            return
        val, decision, test_adj, info, name, weights, oof, test = best
        # новые сиды делают средние семейств надёжнее, но их OOF не обязан быть лучше ранней смеси, которая
        # выбрана по тем же OOF: ответ обновляется, если он не хуже лучшего за прогон больше чем на 0.3%
        top = state["best_val"]
        if state.get("written") == (name, weights):
            return  # тот же кандидат, что уже в файле
        if top is not None and not better(val, top, spec.metric) and abs(val - top) > 0.003 * abs(top) + 1e-12:
            log(f"{note}: лучший кандидат {name} ({spec.metric}={val:.5f}) заметно хуже лучшего ответа "
                f"({top:.5f}) — файл не меняется")
            return
        if spec.task_type == "regression":
            writer.write(test_adj, train_y=y, note=f"{note}: {name}; OOF {spec.metric}={val:.5f}; веса {weights}")
        else:
            on_test = None
            if spec.output_kind == "label":
                try:
                    # тест заметно отличается от train — правило под распределение самого теста
                    on_test = decide_on_test(spec, y, mask, oof, test_adj, decision, state["label_shift"],
                                             shift.get("auc"), shift.get("adv"))
                except Exception:
                    log_exc("decide_on_test")
            if on_test is not None:
                decision, test_adj = on_test
                info = {**info, "test_rule": {"expected_test": decision["expected_test"],
                                              "expected_test_oof_rule": decision["expected_test_oof_rule"],
                                              "class_weights": [round(x, 3) for x in decision["weights"]]}}
            writer.write(test_adj, decision=decision, note=f"{note}: {name}; OOF {spec.metric}={val:.5f}; веса {weights}")
        state["written"] = (name, weights)
        state["written_weights"] = dict(weights) if isinstance(weights, dict) else {}
        if top is None or better(val, top, spec.metric):
            state["best_val"] = val
            state["best_oof"] = oof
            state["best_test"] = test
        try:
            np.save(work / "blend_oof.npy", oof)
            np.save(work / "blend_test.npy", test)
        except Exception:
            pass
        save_json(work / "final.json", {"oof_score": val, "choice": name, "weights": weights, "info": info,
                                        "models": [{"name": r.name, "score": r.score, "iters": r.iters,
                                                    "seconds": round(r.seconds, 1)} for r in results if r]})

    def left():
        return budget.left() - reserve

    big = len(X) > 300_000
    lr_main = 0.05 if big else 0.03
    # 4.1 основная LightGBM с найденной конфигурацией
    lim = max(60.0, min(left() * 0.5, eval_t * 15 + 300))
    # итераций до ранней остановки примерно обратно шагу: если по времени экспериментов видно, что модель с
    # мелким шагом не успеет все фолды, шаг крупнее (недоученный фолд хуже модели с крупным шагом)
    one_fit = eval_t / max(1, researcher.n_seeds) * (1.3 if spec.task_type == "multiclass" else 1.2)
    for lr_try in (lr_main, 0.05, 0.08, researcher.eval_lr):
        lr_main = max(lr_main, lr_try)
        if one_fit * researcher.eval_lr / lr_main * 1.5 <= lim:
            break
    if lr_main != (0.05 if big else 0.03):
        log(f"lgb_main: шаг {lr_main} — с меньшим не успеет за {lim:.0f}s")
    r = safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg, N_JOBS, lim, lr=lr_main, es=200, seed=0, name="lgb_main")
    if r is not None:
        log(f"lgb_main: {spec.metric}={r.score:.5f} итерации={r.iters} за {r.seconds:.0f}s")
        results.append(r)
        publish("lgb_main")
    # 4.1б та же конфигурация с шагом из исследования: дёшево и иногда лучше на сдвинутых данных
    if left() > eval_t * 3 + 60:
        r = safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg, N_JOBS, max(60.0, left() * 0.2), lr=researcher.eval_lr,
                     es=50, seed=7, name="lgb_fast")
        if r is not None and r.test is not None:
            log(f"lgb_fast: {spec.metric}={r.score:.5f} за {r.seconds:.0f}s")
            results.append(r)
            publish("lgb_fast")
    # 4.1в смесь функций потерь для MAE-задач: Huber-LightGBM и L1-LightGBM (всегда обе потери для разнообразия)
    is_mae_task = spec.task_type == "regression" and spec.metric in ("mae", "medae", "smape")
    if is_mae_task and left() > min(90.0, eval_t * 2 + 30):
        if cfg.objective_variant != "huber":
            cfg_alt = ModelConfig(**{**cfg.to_dict(), "objective_variant": "huber"})
            alt_name = "lgb_huber"
        else:
            cfg_alt = ModelConfig(**{**cfg.to_dict(), "objective_variant": None})
            alt_name = "lgb_l1"
        r = safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg_alt, N_JOBS, max(60.0, left() * 0.40), lr=lr_main,
                     es=200, seed=13, name=alt_name)
        if r is not None and r.test is not None:
            log(f"{alt_name}: {spec.metric}={r.score:.5f} итерации={r.iters} за {r.seconds:.0f}s")
            results.append(r)
            publish(alt_name)

    top_cols = None
    imp_src = next((r.importance for r in results if r is not None and r.importance is not None), None)
    if imp_src is None and researcher.best is not None:
        imp_src = researcher.best.importance
    if imp_src is not None:
        top_cols = list(imp_src.sort_values(ascending=False).index)

    # 4.2 Разнообразие семейств — СНАЧАЛА все 7 быстрых/сильных базовых семейств (Ridge/LogReg, CatBoost, XGB-PH, HGB, ET, MLP),
    # и только ЗАТЕМ вторичные модели (lgb_et, xgb, cat_top, huber, lasso), чтобы при сжатом бюджете ансамбль всегда был полным!
    rich_cats = [c for c in prep.cat_cols if c in X.columns and X[c].nunique() >= 30]  # флаги и короткие списки не в счёт
    late_cat = spec.task_type == "multiclass" and len(spec.classes) > 3 and len(rich_cats) < 3
    mc_many = spec.task_type == "multiclass" and len(spec.classes) >= 10
    mc_wide = spec.task_type == "multiclass" and len(spec.classes) >= 6
    xgb_lr = 0.1 if big else (0.08 if mc_wide else 0.04)
    xgb_depth = 6 if (X.shape[1] > 100 or mc_wide) else 7
    cat_kw = {"cat_cols": prep.cat_cols, "depth": 5 if mc_many else 6,
              "lr": 0.12 if (big or (spec.task_type == "multiclass" and len(spec.classes) > 3)) else 0.06}
    cfg_x = cfg
    if spec.task_type == "regression" and cfg.target_transform == "log1p" and spec.metric in ("rmse", "mse", "r2"):
        cfg_x = ModelConfig(**{**cfg.to_dict(), "target_transform": "none"})

    def fit_extra(name, fn, kw, min_need=None, use_clean=False):
        need = min(80.0, eval_t * 2 + 25) if min_need is None else min_need
        if left() < need:
            log(f"{name}: пропуск, мало времени ({left():.0f}s)")
            return
        lim = max(30.0 if min_need is not None else 55.0, left() * 0.45)
        kwargs = dict(kw)
        Xf, Xtf = (prep.X, prep.Xt) if use_clean else (X, Xt)
        cf = ModelConfig(**{**cfg_x.to_dict(), "drop": []}) if use_clean else cfg_x
        if fn is fit_cat:
            r = safe_fit(fn, Xf, y, Xtf, plan, spec, cf, kwargs.pop("cat_cols"), N_JOBS, lim, name=name, **kwargs)
        else:
            r = safe_fit(fn, Xf, y, Xtf, plan, spec, cf, N_JOBS, lim, name=name, **kwargs)
        if r is not None and r.test is not None:
            log(f"{name}: {spec.metric}={r.score:.5f} итерации={r.iters} за {r.seconds:.0f}s")
            results.append(r)
            publish(name)

    # Tier 1: базовые разнообразные семейства (все видят принятые AutoResearch-признаки: text_oof, entity_history, LLM)
    if spec.task_type == "regression":
        fit_extra("ridge", fit_linear, {"kind": "ridge", "top_cols": top_cols}, min_need=15.0)
    else:
        fit_extra("logreg", fit_linear, {"kind": "logreg", "top_cols": top_cols}, min_need=20.0)
    if not late_cat:
        fit_extra("cat", fit_cat, cat_kw, min_need=40.0)
        if is_mae_task:
            fit_extra("cat_rmse", fit_cat, {**cat_kw, "loss_override": "RMSE", "seed": 2}, min_need=40.0)
    if is_mae_task:
        fit_extra("xgb_ph", fit_xgb, {"lr": 0.04 if not big else 0.1, "depth": 6, "objective_override": "pseudohuber", "seed": 11}, min_need=40.0)
    else:
        fit_extra("xgb", fit_xgb, {"lr": xgb_lr, "depth": xgb_depth}, min_need=40.0)
    fit_extra("hgb", fit_hgb, {"top_cols": top_cols}, min_need=25.0)
    fit_extra("et", fit_et, {}, min_need=25.0)
    fit_extra("mlp", fit_mlp, {"top_cols": top_cols, "prep_mode": "qt"}, min_need=25.0)
    # Второе нейросетевое семейство Dual-View [Quantile + Standard] + дополнительные сиды при быстром обучении
    mlp_r = next((r for r in results if r is not None and r.name == "mlp" and r.test is not None), None)
    if mlp_r is not None and mlp_r.seconds <= 90.0 and left() > 140.0:
        fit_extra("mlp_dual", fit_mlp, {"top_cols": top_cols, "prep_mode": "dual", "seed": 100}, min_need=30.0)
    if mlp_r is not None and mlp_r.seconds <= 75.0 and left() > 180.0:
        n_extra_mlp = 3 if (mlp_r.seconds <= 45.0 and left() > 360.0) else 2
        for ms in range(1, n_extra_mlp + 1):
            if left() <= 120.0:
                break
            fit_extra(f"mlp_s{ms}", fit_mlp, {"top_cols": top_cols, "prep_mode": "qt", "seed": ms}, min_need=35.0)
        md_r = next((r for r in results if r is not None and r.name == "mlp_dual" and r.test is not None), None)
        if md_r is not None and md_r.seconds <= 60.0 and left() > 240.0:
            for ms in range(1, 3):
                if left() <= 140.0:
                    break
                fit_extra(f"mlp_dual_s{ms}", fit_mlp, {"top_cols": top_cols, "prep_mode": "dual", "seed": 100 + ms}, min_need=35.0)

    # Tier 2: вторичные модели для усиления ансамбля (когда все базовые семейства уже обучены)
    if left() > min(90.0, eval_t * 2 + 30):
        cfg_et = ModelConfig(**{**cfg.to_dict(), "lgb_params": {**cfg.lgb_params, "extra_trees": True}})
        r = safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg_et, N_JOBS, max(60.0, left() * 0.35), lr=lr_main,
                     es=200, seed=19, name="lgb_et")
        if r is not None and r.test is not None:
            log(f"lgb_et: {spec.metric}={r.score:.5f} итерации={r.iters} за {r.seconds:.0f}s")
            results.append(r)
            publish("lgb_et")
    if is_mae_task:
        fit_extra("xgb", fit_xgb, {"lr": 0.04 if not big else 0.1, "depth": 6 if X.shape[1] > 100 else 7}, min_need=45.0)
    if not late_cat and top_cols and X.shape[1] > 350 and left() > 90.0:
        # На широких БД (>350 колонок): дополнительный CatBoost на очищенном top-250 (меньше шума, быстрее)
        keep250 = [c for c in top_cols if c in X.columns][:250]
        if len(keep250) >= 80:
            drop_rest = [c for c in X.columns if c not in set(keep250)]
            cfg_top = ModelConfig(**{**cfg_x.to_dict(), "drop": sorted(set(cfg_x.drop) | set(drop_rest))})
            r = safe_fit(fit_cat, X, y, Xt, plan, spec, cfg_top, prep.cat_cols, N_JOBS,
                         max(55.0, left() * 0.4), lr=cat_kw["lr"], depth=7, seed=5, name="cat_top")
            if r is not None and r.test is not None:
                log(f"cat_top: {spec.metric}={r.score:.5f} итерации={r.iters} за {r.seconds:.0f}s")
                results.append(r)
                publish("cat_top")
    if spec.task_type == "regression":
        fit_extra("huber", fit_linear, {"kind": "huber", "top_cols": top_cols}, min_need=25.0)
        fit_extra("lasso", fit_linear, {"kind": "lasso", "top_cols": top_cols}, min_need=25.0)

    # 4.2б регрессия с историей объекта: модель-поправка к прошлому значению (после базовых семейств)
    hist_cols = [c for c in X.columns if c.startswith("hist__last")] if spec.task_type == "regression" else []
    offsets = [(f"lgb_offset{j}", oc, X, Xt) for j, oc in enumerate(hist_cols[:2])]
    if hist_cols and plan.time_col:
        for key in extra_entity_keys(researcher, 3 - len(offsets)):
            key = [c for c in key if c in prep.X.columns and c != plan.time_col]
            if not key:
                continue
            try:
                t_tr, t_te = entity_periods(prep.X, prep.Xt, plan.time_col)
                ha, hb = op_entity_history(prep.X, prep.Xt, y, key, plan.time_col, t_tr, t_te)
                tag = f"__k{len(offsets)}"
                ha, hb = ha.add_suffix(tag).reset_index(drop=True), hb.add_suffix(tag).reset_index(drop=True)
                offsets.append((f"lgb_xoffset{len(offsets)}", "hist__last" + tag, pd.concat([X, ha], axis=1),
                                pd.concat([Xt, hb], axis=1)))
                log(f"поправка и к истории по ключу {key} (в исследовании не принят)")
            except Exception:
                log_exc("extra entity key")
    for nm, oc, Xo, Xto in offsets:
        if left() < eval_t * 3 + 60:
            break
        for sd in (0, 1):
            if left() < eval_t * 3 + 60:
                break
            r = safe_fit(fit_lgb, Xo, y, Xto, plan, spec, cfg, N_JOBS, max(60.0, left() * 0.25), lr=lr_main, es=200,
                         seed=sd, name=f"{nm}_s{sd}", offset_col=oc)
            if r is not None and r.test is not None:
                log(f"{r.name} (поправка к {oc}): {spec.metric}={r.score:.5f} за {r.seconds:.0f}s")
                results.append(r)
                publish(r.name)
    # 4.3 ещё сиды, пока есть время: сиды одного семейства усредняются до смешивания (см. publish),
    # поэтому каждый новый сид уменьшает разброс модели, а не добавляет шанс случайно удачного сида
    def lgb_seed(k):
        cfg2 = ModelConfig(**{**cfg.to_dict()})
        p = dict(cfg2.lgb_params)
        p["feature_fraction"] = max(0.3, p.get("feature_fraction", 0.7) * 0.85)
        p["bagging_fraction"] = 0.75
        cfg2.lgb_params = p
        return [safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg2, N_JOBS, max(60.0, left() * 0.5), lr=lr_main, es=200,
                         seed=k, name=f"lgb_s{k}")]

    def xgb_seed(k):
        depths = (6, 7, 5, 7, 6)
        return [safe_fit(fit_xgb, X, y, Xt, plan, spec, cfg_x, N_JOBS, max(60.0, left() * 0.45),
                         lr=xgb_lr, depth=min(depths[k % len(depths)], 6) if (mc_wide or X.shape[1] > 100) else depths[k % len(depths)],
                         seed=k, name=f"xgb_s{k}")]

    def offset_seed(k):
        return [safe_fit(fit_lgb, Xo, y, Xto, plan, spec, cfg, N_JOBS, max(60.0, left() * 0.25), lr=lr_main, es=200,
                         seed=k + 1, name=f"{nm}_s{k + 1}", offset_col=oc)
                for nm, oc, Xo, Xto in offsets]

    def et_seed(k):
        return [safe_fit(fit_et, X, y, Xt, plan, spec, cfg_x, N_JOBS, max(60.0, left() * 0.45), seed=k,
                         name=f"et_s{k}")]

    def cat_seed(k):
        depths = (6, 7, 6, 7, 5)
        kw_k = {**cat_kw, "depth": depths[k % len(depths)]}
        return [safe_fit(fit_cat, X, y, Xt, plan, spec, cfg_x, kw_k.pop("cat_cols"), N_JOBS,
                         max(60.0, left() * 0.45), seed=k, name=f"cat_s{k}", **kw_k)]

    def cat_rmse_seed(k):
        depths = (6, 7, 6, 5)
        kw_k = {**cat_kw, "depth": depths[k % len(depths)], "loss_override": "RMSE"}
        return [safe_fit(fit_cat, X, y, Xt, plan, spec, cfg_x, kw_k.pop("cat_cols"), N_JOBS,
                         max(60.0, left() * 0.45), seed=20 + k, name=f"cat_rmse_s{k}", **kw_k)]

    def xgb_ph_seed(k):
        depths = (6, 7, 5)
        return [safe_fit(fit_xgb, X, y, Xt, plan, spec, cfg_x, N_JOBS, max(60.0, left() * 0.45),
                         lr=0.04 if not big else 0.1, depth=depths[k % len(depths)],
                         objective_override="pseudohuber", seed=30 + k, name=f"xgb_ph_s{k}")]

    def lgb_huber_seed(k):
        cfg2 = ModelConfig(**{**cfg.to_dict(), "objective_variant": "huber"})
        p = dict(cfg2.lgb_params)
        p["feature_fraction"] = max(0.3, p.get("feature_fraction", 0.7) * 0.85)
        p["bagging_fraction"] = 0.75
        cfg2.lgb_params = p
        return [safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg2, N_JOBS, max(60.0, left() * 0.4), lr=lr_main, es=200,
                         seed=100 + k, name=f"lgb_huber_s{k}")]

    def lgb_l1_seed(k):
        cfg2 = ModelConfig(**{**cfg.to_dict(), "objective_variant": None})
        p = dict(cfg2.lgb_params)
        p["feature_fraction"] = max(0.3, p.get("feature_fraction", 0.7) * 0.85)
        p["bagging_fraction"] = 0.75
        cfg2.lgb_params = p
        return [safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg2, N_JOBS, max(60.0, left() * 0.4), lr=lr_main, es=200,
                         seed=120 + k, name=f"lgb_l1_s{k}")]

    def hgb_seed(k):
        return [safe_fit(fit_hgb, X, y, Xt, plan, spec, cfg_x, N_JOBS, max(60.0, left() * 0.35),
                         top_cols=top_cols, seed=40 + k, name=f"hgb_s{k}")]

    def mlp_seed(k):
        return [safe_fit(fit_mlp, X, y, Xt, plan, spec, cfg_x, N_JOBS, max(60.0, left() * 0.3),
                         top_cols=top_cols, prep_mode="qt", seed=10 + k, name=f"mlp_s{10 + k}")]

    def mlp_dual_seed(k):
        return [safe_fit(fit_mlp, X, y, Xt, plan, spec, cfg_x, N_JOBS, max(60.0, left() * 0.3),
                         top_cols=top_cols, prep_mode="dual", seed=110 + k, name=f"mlp_dual_s{10 + k}")]

    def lgb_et_seed(k):
        cfg2 = ModelConfig(**{**cfg.to_dict(), "lgb_params": {**cfg.lgb_params, "extra_trees": True}})
        return [safe_fit(fit_lgb, X, y, Xt, plan, spec, cfg2, N_JOBS, max(60.0, left() * 0.35), lr=lr_main, es=200,
                         seed=60 + k, name=f"lgb_et_s{k}")]

    # Вес-ориентированный отбор семейств для мультисидирования: доучиваем сиды КАЖДОМУ семейству с весом >= 5% в бленде
    # или близкому к лидеру (<= 5%), упорядочивая по текущему весу в ансамбле!
    done = [r for r in results if r is not None and r.test is not None and np.isfinite(r.score)]
    top = max(done, key=lambda r: r.score if direction(spec.metric) == "max" else -r.score).score if done else None
    cur_w = state.get("written_weights") or {}
    fam_w = {}
    for wk, wv in cur_w.items():
        base_f = re.sub(r"x\d+$", "", str(wk))
        f = model_family(base_f)
        fam_w[f] = fam_w.get(f, 0.0) + float(wv)
    fam_best = {}
    for r in done:
        f = model_family(r.name)
        if f not in fam_best or better(r.score, fam_best[f], spec.metric):
            fam_best[f] = r.score

    cand_map = [
        ("cat", cat_seed, not late_cat),
        ("cat_rmse", cat_rmse_seed, not late_cat),
        ("lgb_huber", lgb_huber_seed, True),
        ("lgb_l1", lgb_l1_seed, True),
        ("lgb", lgb_seed, True),
        ("lgb_et", lgb_et_seed, True),
        ("xgb_ph", xgb_ph_seed, True),
        ("xgb", xgb_seed, True),
        ("hgb", hgb_seed, True),
        ("et", et_seed, True),
        ("mlp", mlp_seed, True),
        ("mlp_dual", mlp_dual_seed, True),
    ]
    picked = []
    for f, fn, allowed in cand_map:
        if not allowed or f not in fam_best:
            continue
        w_f = fam_w.get(f, 0.0)
        close = top is not None and abs(fam_best[f] - top) <= 0.05 * max(abs(top), 1e-9)
        if f == "lgb" or w_f >= 0.05 or close:
            picked.append((w_f, 1 if close else 0, fn))
    picked.sort(key=lambda x: (-x[0], -x[1]))
    seeders = [fn for _, _, fn in picked] or [lgb_seed]
    if any(r is not None and r.name.startswith(("lgb_offset", "lgb_xoffset")) for r in results):
        seeders.append(offset_seed)
    next_k = {fn: 1 for fn in seeders}
    # на быстрых данных сидов больше: семейство поправки на WiDS из 14 моделей 33.76, из 28 — 33.58; на медленных
    # данных раньше кончится время
    max_seeds = 30

    def seed_rounds(frac_limit):
        # семейства по очереди, у каждого свой номер сида; после доли бюджета frac_limit новые сиды не начинаются
        while any(k <= max_seeds for k in next_k.values()):
            for fn in seeders:
                k = next_k[fn]
                if k > max_seeds:
                    continue
                if not (left() > max(eval_t * 4, 120) and (k <= 3 or budget.frac_used() < frac_limit)):
                    return
                for r in fn(k):
                    if r is not None and r.test is not None:
                        log(f"{r.name}: {spec.metric}={r.score:.5f} за {r.seconds:.0f}s")
                        results.append(r)
                        publish(r.name)
                next_k[fn] = k + 1

    seed_rounds(0.65 if late_cat else 1.0)
    if late_cat:
        fit_extra("cat", fit_cat, cat_kw)
        # CatBoost пропущен или закончил раньше — оставшееся время снова на сиды (UNSW-NB15: иначе ~45 минут простоя)
        seed_rounds(1.0)
    # Zero-aware MAPE hurdle: при наличии точных нулей в таргете обнуляем прогноз там, где P(y=0) высока
    if (spec.task_type == "regression" and spec.metric == "mape" and state.get("best_oof") is not None
            and state.get("best_test") is not None and (np.abs(y) < 1e-12).mean() >= 0.01):
        try:
            hur = apply_zero_mape_hurdle(X, Xt, y, state["best_oof"], state["best_test"], plan, mask, n_jobs=N_JOBS)
            if hur is not None and better(hur["score"], state["best_val"], spec.metric):
                writer.write(hur["test"], train_y=y,
                             note=f"zero-MAPE hurdle (thr={hur['threshold']:.4f}); OOF mape={hur['score']:.5f}")
                state["best_val"] = hur["score"]
                save_json(work / "final.json", {
                    "oof_score": hur["score"], "choice": "zero_mape_hurdle",
                    "weights": state.get("written_weights", {}), "info": {"threshold": hur["threshold"]},
                    "models": [{"name": r.name, "score": r.score, "iters": r.iters,
                                "seconds": round(r.seconds, 1)} for r in results if r],
                })
        except Exception:
            log_exc("apply_zero_mape_hurdle")
    if not results:
        log("финальные модели не обучились — остаётся предыдущий submission")


def write_report(work: Path, spec, plan, researcher, llm):
    try:
        lines = [f"# Отчёт агента: {Path(spec.task_path).name}", "",
                 f"- Тип задачи: {spec.task_type}; метрика: {spec.metric} (источник: {spec.metric_source})",
                 f"- Таргет: `{spec.target}`; id: `{spec.id_col}`; формат ответа: {spec.output_kind}",
                 f"- Валидация: {plan.desc}", f"- LLM: {llm.stats()}", "", "## Журнал экспериментов", "",
                 "| # | Источник | Гипотеза | Скор | Δ | Принята |", "|---|---|---|---|---|---|"]
        for e in researcher.journal:
            if e.get("kind") == "note":
                continue
            s = e.get("score")
            d = e.get("delta")
            lines.append(f"| {e['id']} | {e.get('source')} | {str(e.get('title')).replace('|', '/')} | "
                         f"{'' if s is None else f'{s:.5f}'} | {'' if d is None else f'{d:+.5f}'} | "
                         f"{'да' if e.get('accepted') else ('нет' if e.get('accepted') is False else '')} |")
        notes = [e for e in researcher.journal if e.get("kind") == "note"]
        if notes:
            lines += ["", "## Анализ LLM", ""] + [f"- {n.get('analysis')}" for n in notes]
        atomic_write_text(work / "report.md", "\n".join(lines))
    except Exception:
        log_exc("write_report")


def main():
    task_path = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    try:
        run(task_path)
    except Exception:
        log_exc("run")
        out = task_path.parent / "submission.csv"
        if not out.exists():
            log("submission.csv так и не появился")
    finally:
        log(f"завершение через {elapsed():.0f}s")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
