# Результаты ветки `exp/rodion-v11-resnet-pools14` (Родион)

- Ветка: `exp/rodion-v11-resnet-pools14` (создана от актуального `origin/main` `dc6ec5f` — **v10**, с полным переносом улучшений пулов 1–4 поверх `v10` без потери ни одной фичи или теста `v9`/`v10` и Pool 5)
- Затронутые файлы: `ag_database.py`, `ag_relational.py`, `ag_features.py`, `ag_models.py`, `ag_ensemble.py`, `ag_worker.py`, `ag_research.py`, `solution.py`, `tools/test_final.py`, `tools/bench_pool1_ram.py`, `tools/analyze_task2_errors.py`, `tools/analyze_task1_signal.py`, `tools/bench_neural_mae.py`, `tools/bench_multiclass.py`, `NOTES_rodion.md`

## Что перенесено поверх актуального `main` (`v10`, `dc6ec5f`)
- **Сохранено 100% из `v9`/`v10` и Pool 5:** `_convert_pg_booleans`, `_unix_epoch_unit` / `_parse_time_series` (Unix epoch + `DD.MM.YYYY`), калиброванный `ag_nowcast.py`, Tier-1/Tier-2 `final_stage` с сортировкой сидов по весу, `lgb_et`, `cat_top`, `apply_zero_mape_hurdle`, `log_odds` симплекс-бленд, `tools/release_check.py`, `tools/stress_private.py` и все 56 тестов `v10`.
- **Пул 1 — Реляционный движок, схемы Private-БД и RAM:**
  - `ag_database.py`: чтение `FOREIGN KEY` через `pg_catalog.pg_constraint` (`contype='f'`) в первую очередь с fallback на `information_schema` (критично для read-only роли `hack_reader`, где `information_schema` может отдавать 0 строк); безопасный `_downcast_df` (`int64->int32`, не-ID `float64->float32`); `SpooledTemporaryFile(max_size=32MB, dir=work)` вместо дефолтного `/tmp`; чтение `COMMENT ON COLUMN` из `pg_catalog.pg_description` + `comments_to_dictionary` при отсутствии таблицы `data_dictionary`.
  - `ag_relational.py`: `__sys_max` в `_temporal_join` на том же завершённом часе; строго каузальный скользящий `nunique` категорий по окнам `1d/7d/30d` (`_causal_window_cat_nunique`) в `_aggregate_child`; эвристические многоролевые джойны (`sender_id`/`receiver_id -> users.id`) при отсутствии SQL FK; запрет `_keys` на джойн по голому `timestamp` при наличии общего ID; фикс утечки `v10` на нерегулярных дочерних таблицах `(entity_id, timestamp)` с одновременными метками (`_is_irregular_timed_table` гарантированно направляет их в строго каузальный `_aggregate_child < event_time`).
- **Пул 2 — Признаки и кросс-табличный матчинг:**
  - `ag_features.py`: хранение `*__ord` строго в `float64` (исключает ~2.8 мин округление `float32` на стыке train/test); ускоренный векторизованный тег-матчинг (`0.497s` на 400k строках); универсальный кросс-табличный категориальный матчинг `__eq__` (пересечение словарей `>= 30%` строго по `train`, пакетная сборка через `pd.concat`).
- **Пул 3 — Zero-dependency L1/Huber/CE нейросеть, быстрый HGB и неотрицательный симплекс-стекинг:**
  - `ag_models.py`: `sklearn.MLPRegressor`/`MLPClassifier` (поддерживавший только L2) заменён на компактный `_NumpyTabularResNet` (`Linear(128)->LN->SiLU->Dropout->Residual(128)->Linear(64)->LN->SiLU->Dropout->Linear(out)`, `Uniform(-1/sqrt(d))`, глобальный L2 `clip_grad_norm_(5.0)`, AdamW, CosineLR, Early Stopping, `_TimeGuard`) с нативными потерями `L1`, `SmoothL1/Huber`, `MSE`, `BCEWithLogits`, `CrossEntropy` и `sample_weight`; `fit_hgb` переведён на пошаговый `warm_start=True` + раннюю остановку + `_TimeGuard` (с сохранением train-only `umap` для object-колонок и `threadpool_limits` из `v10`); добавлены ускорения мультикласса в `fit_cat` (`border_count=128` при `>= 6` классах) и `fit_xgb` (`depth<=5, max_bin=128, colsample_bytree=0.5` при `>= 10` классах).
  - `ag_ensemble.py`: в `stack_blend` для классификации `LogisticRegression` заменён на неотрицательную оптимизацию на симплексе (`w >= 0, sum(w) = 1`) напрямую под целевую метрику (`L-BFGS-B` с аналитическим градиентом для `logloss`, быстрый `_simplex_cls_weights` для `PR-AUC`/`ROC-AUC`) с поддержкой `>= 2` временных фолдов; в `_simplex_reg_weights` добавлен wall-clock лимит `time_budget=4.0s` и `maxfev`.
  - `solution.py`: `comments_to_dictionary` fallback, адаптивные `cat_kw`/`xgb_lr`/`xgb_depth` на мультиклассе и ограничение глубины `xgb_seed <= 6` на широких таблицах (`>100` признаков).
- **Пул 4 — AutoResearch, 2-ступенчатый poison-фильтр и бюджет:**
  - `ag_worker.py`: объединённый 2-ступенчатый poison-фильтр (отравление будущих строк `test` + онлайн-усечение) с точечным отбрасыванием только протекающих колонок (`POISON_DROP`) и полным отказом с ошибкой `temporal leakage detected` при утечке всех колонок.
  - `ag_research.py` + `solution.py`: анти-утечное правило в `repair_code()`, ранний выход после 2 раундов LLM подряд без прироста (`AGENT_LLM_EARLY_EXIT=1`), запрет дублирующего `retry` без смены конфигурации (`_retry_sig`), `time_breakdown` в `journal.json`, настраиваемый сплит `AGENT_RESEARCH_FRAC=0.30`.

## Что сработало (с цифрами против `main` v8/v10)
- **Task 2 (`hack_final_task2`, 3-fold хронологическая CV по `timestamp__ord`, MAE, строго `<= 09:00`):**
  - Ориентиры: `main v8 (6b499e3)`: `CatBoost = 2.162`, `LGB = 2.273`; `v10 (dc6ec5f)` с `sklearn` L2 MLP: `mlp ~ 2.35`, бленд `~2.12–2.13`.
  - **Новый замер `v11` (`exp/rodion-v11-resnet-pools14`, `v10` calibrated nowcast + `_NumpyTabularResNet` L1 + GBDTs, `674` признака, `timestamp__ord` `float64`):**
    - `lgb_huber`: `2.20681` (`94.7s`)
    - `cat` (MAE): `2.17752` (`116.0s`)
    - `cat_rmse` (RMSE): **`2.17011`** (`93.1s`)
    - `xgb_ph` (PseudoHuber): `2.27393` (`44.7s`)
    - `hgb` (warm_start L1): `2.23130` (`40.4s`)
    - `mlp_s0 = 2.12429` (`28.7s`), `mlp_s1 = 2.09411` (`29.5s`), `mlp_s2 = 2.18441` (`24.4s`) -> **`mlpx3 = 2.09764`** (`82.5s`)
    - **Симплекс-ансамбль (`V11_BLEND`):** **`2.06668`** (`mlpx3: 0.622`, `cat_rmse: 0.279`, `cat: 0.077`, `hgb: 0.023`) — **−0.095 MAE** против `main v8` (`2.162`) и **~−0.06 MAE** против `v10`!
    - **Вложенный 3-fold OOF-стекинг (`V11_STACK`):** **`2.06991`** (все веса `>= 0`).
- **Стресс-стенд Private-БД (`tools/stress_private.py --primary-only --budget 60 --rows 2000 --large-rows 200000 --threads 4`):** **5 / 5 PASS**, `0` ошибок, `0` watchdog:
  - `fraud` (`PR-AUC`): `PASS`, `PR-AUC=0.529`, `ROC-AUC=0.879`, `0.24 GiB`, `34.6s` (`logreg: 0.842`, `mlp: 0.154`).
  - `multiclass` (`LogLoss`, 12 классов + сдвиг долей): `PASS`, `LogLoss=0.378`, `F1-macro=0.862`, `0.25 GiB`, `37.2s` — `_NumpyTabularResNet` соло дал `0.357` против `0.420` у `lgb_main`, получил вес `0.635` и довёл OOF LogLoss смеси до **`0.343`**.
  - `regression` (`RMSLE`, 18% нулей + грязные строки): `PASS`, `RMSLE=0.966`, `0.23 GiB`, `28.4s`.
  - `text` (русские тексты + временной сплит, `PR-AUC`): `PASS`, `PR-AUC=1.0`, `future_text_poison_delta=0.0`, `0.99 GiB`, `32.5s`.
  - `large` (`200k` строк, `MAE`): `PASS`, `MAE=0.794`, `0.35 GiB`, `38.7s`.
- **Замеры пулов 1–4 из серии экспериментов (подтверждены на сервере 8 CPU / 31 ГБ RAM):**
  - **Пул 1 RAM (2M child -> 50k events, `tools/bench_pool1_ram.py`):** `SpooledTemporaryFile` RSS `0.143 GB`, `_downcast_df` `0.268 -> 0.246 GB`, `enrich` `13.74s`, общий пик RSS **`1.51 GB`**.
  - **Пул 2 (Task 1 400k + Task 2 error decomposition):** скорость матчинга на Task 1 **`0.497s`** (`< 5s`), `48.8%` ошибок `>5 мин` на Task 2 приходятся на переходы шторма (`6.75%` строк).
  - **Пул 3 (`torch==2.5.1+cpu` vs NumPy ResNet, 15-class multiclass):** `torch` L1 `2.1354` требует `+41s` и `+807 MB`, тогда как zero-dependency `_NumpyTabularResNet` дал `2.094–2.137` при `0s` установки; на 15 классах × 50k все 7 семейств укладываются в тайм-лимиты (`180s` и `10s`).
  - **Пул 4 (длинные 40-минутные E2E с `Qwen3.8-27B`):** Task 2 `30/70 + early-exit` дал OOF MAE **`2.07287`** (`3.43 GB` RAM, экономия `343.5s` = 5.7 мин на AutoResearch, `0` poison-дропов) против `2.07316` у `45/55`; Task 1 `45/55` дал OOF PR-AUC **`0.15472`** (`3.31 GB` RAM, `0` poison-дропов).

## Что не сработало
- Предвестники `07:00 -> 08:00 -> 09:00` (`d2` + наклон 3ч) ухудшали `lgb_huber` (`+0.003` MAE) — не добавлялись глобально.
- `CatBoost Huber` на Task 2 дал `MAE = 3.31` — не используется (оставлены `MAE` и `RMSE`).
- Добавление `torch==2.5.1+cpu` в `requirements.txt` признано избыточным из-за `+41s`/`807 MB` и риска `--index-url` при равном/лучшем качестве `_NumpyTabularResNet`.

## Проверка регламента
- Утечек будущего нет (`<= 09:00` / `< t`): **да** (`tools/release_check.py` `poison_audit` на реальной БД `hack_final_task2`: `sensors: 0/526`, `events: 0/526`, `combined: 0/526`; `stress_private` `future_text_poison_delta = 0.0`; 2-ступенчатый poison-фильтр в `ag_worker.py`; защита нерегулярных `(entity_id, timestamp)` таблиц в `ag_relational.py`).
- Хардкода под конкретные таблицы/колонки нет: **да**.
- Новые пакеты в `requirements.txt`: **нет** (ровно 14 разрешённых пакетов из белого списка без изменений).
- Тесты и плоский ZIP: **passed** (`py_compile` OK; **`63 / 63 OK`** за `20.1s` в `tools/test_final.py`; `tools/stress_private.py` **5/5 PASS**; `dist/solution_v11.zip` прошёл `audit_zip` — `14` файлов, `126,638` байт, `SHA-256: 1c7e37c4b7b8e3c1a32699c4f091c9aef966d6af38505db65ce99051ca6cecfc`, 0 подкаталогов, 0 CRLF, 0 секретов, 0 `sys.path`/`PYTHONPATH`).
