# Результаты ветки `exp/rodion-v11-resnet-pools14` (Родион)

- Ветка: `exp/rodion-v11-resnet-pools14` (создана от `origin/main` `dc6ec5f` — **v10**, с переносом пулов 1–4 и полным ночным циклом улучшений без потери ни одной фичи или теста `v9`/`v10` и Pool 5)
- Затронутые runtime-файлы: `ag_database.py`, `ag_relational.py`, `ag_features.py`, `ag_nowcast.py`, `ag_models.py`, `ag_ensemble.py`, `ag_worker.py`, `ag_research.py`, `solution.py`
- Тесты и бенчмарки: `tools/test_final.py` (`69` тестов), `tools/bench_pool1_ram.py`, `tools/analyze_task2_errors.py`, `tools/analyze_task1_signal.py`, `tools/bench_neural_mae.py`, `tools/bench_multiclass.py`, `NOTES_rodion.md`

## Ключевые улучшения поверх `origin/main` (`v10`, `dc6ec5f`)

### 1. Реляционный движок, схемы Private-БД и RAM (`ag_database.py`, `ag_relational.py`)
- `pg_catalog.pg_constraint` (`contype='f'`) с fallback на `information_schema` (read-only роли), безопасный `_downcast_df`, `SpooledTemporaryFile(max_size=32MB, dir=work)`, `pg_description` -> `data_dictionary` (`comments_to_dictionary`).
- `__sys_max` в `_temporal_join`, строго каузальный скользящий `nunique` категорий (`1d/7d/30d`), автоматические высокочастотные окна `1h`/`6h` + `hist_burst_1h_1d` для подсуточных дочерних потоков в `_aggregate_child`.
- Единый `POST_FACTUM_RE`-фильтр во всех путях джойна (SQL FK, эвристика, интервалы, дочерние таблицы), эвристические многоролевые джойны (`sender_id`/`receiver_id -> users.id`), запрет джойна по голому `timestamp` при наличии ID и защита от утечки на нерегулярных `(entity_id, timestamp)` таблицах (`_is_irregular_timed_table`).

### 2. Признаки, реконструкция сбоев и каузальный Nowcaster v2 (`ag_features.py`, `ag_nowcast.py`)
- Хранение `*__ord` в `float64`, быстрый векторизованный тег-матчинг (`0.497s` на 400k строк), кросс-табличный категориальный матчинг `__eq__` (словарь строго по `train`), байесовски сглаженная каузальная история объекта `hist__smooth` (`m=20`, `< t`).
- Исправлен `cross_impute_outages`: добавлены `__sys_max`, `__ord`, `__year` в `_lag_sfx` (реконструкция сокращена с `91` до `51` реальных сенсорных колонок, убрана линейная экстраполяция времени в test).
- Исправлен `entity_key_candidates`: исключены `__eq__` и джойненные колонки с `__`, составные ключи ограничены `<= 3–4` колонками (полностью устранены 10–37-колоночные мусорные ключи `entity_history`, просаживавшие `mlp` на `+0.05–0.06 MAE`).
- **Nowcaster v2 (`ag_nowcast.py`):** добавлена строго каузальная аномалия `lag1_anom = lag1(g, h-1) - prof_all(g, h-1, w)` (через backward `prev_idx` без пересечения групп/пропусков) и покалоночный подбор `alpha` Ridge из `(25, 100, 300, 1000, 3500, 12000)` по точному GCV через одно спектральное разложение `eigh(G)`.

### 3. Табличная нейросеть SwiGLU-ResNet + Wide&Deep Skip + RealMLP Gate и ускорение моделей (`ag_models.py`, `solution.py`)
- Zero-dependency `_NumpyTabularResNet` переведён на архитектуру **SwiGLU-ResNet + Wide&Deep линейный skip (`x @ Ws`) + обучаемый гейт масштаба признаков (`x = x_raw * s_in`)** с адаптивной регуляризацией, 3-epoch warmup + cosine LR, bias-corrected EMA, Top-3 SWA, стратифицированным внутренним early-stopping сплитом для `bce`/`ce`, батч-инференсом (`batch_size=8192`, чанки `65_536` в `pred_one` — снижение пикового RAM инференса с `~3.8 GB` до `~50 MB` и масштабирование до `1.2M` строк), 2-бэговым темпоральным `refit` и поддержкой L1 (`mae`/`medae`/`smape`/`mape`), Huber, MSE, BCE, CrossEntropy.
- Разделены признаковые виды двух нейросетевых семейств: `mlp` (`prep_mode="qt"`, L1 на MAE) работает на чистых базовых + nowcast + `text_oof` признаках, `mlp_dual` (`prep_mode="dual"`, гладкий Huber на MAE) — на полном `X` со всеми принятыми AutoResearch-признаками.
- **`cat_top` (Tier 1 на широких БД `> 200` колонок):** CatBoost на `top-150` признаках (`depth=7`, RMSE на MAE-задачах) + `cat_top_seed` в `seeders` (`−0.056 MAE` и почти `2x` быстрее полного `cat_rmse`).
- Пошаговый `warm_start` `fit_hgb` с `_TimeGuard`, ускорение мультикласса в `fit_cat`/`fit_xgb`, ограничение `fit_et` и `fit_mlp` на крупных таблицах (`>120k–150k` строк), быстрая отсечка бесполезных `lgb_offset`-моделей (`>3–5%` хуже лидера), адаптивный `min_need` в `fit_extra` для коротких бюджетов.

### 4. Ансамблирование, AutoResearch и защита от утечек (`ag_ensemble.py`, `ag_research.py`, `ag_worker.py`)
- Неотрицательный симплекс-стекинг (`w >= 0, sum(w) = 1`) для регрессии и классификации; точная векторизованная метрика `_fast_bin_rank_score` (`~5x` быстрее sklearn с точностью `1e-11`), стратифицированная подвыборка (`>25k`), покоординатный поиск + active-set Powell в `_simplex_cls_weights` (устранено схлопывание в равномерные веса `1/k` на крупных PR-AUC задачах; лимит семейств расширен до `24`).
- Защита `apply_zero_mape_hurdle` от вырожденных константных нулей (`>= 2%` OOF / `>= 1%` test ненулевых и `ptp > 1e-8`).
- `error_analysis()` для бинарных `PROBA_METRICS` выводит `pos_rate`, квантили, `top-10% lift`, `top-5% precision` и слабейшие категориальные сегменты (отсортированные по важности модели), для MAE — медиану остатка и смещение; 2-ступенчатый poison-фильтр в `ag_worker.py`, ранний выход после 2 раундов без прироста и анти-утечное правило в `repair_code()`.

## Подтверждённые результаты на сервере

### 1. Task 2 (`hack_final_task2`, 3-fold хронологическая CV, MAE, строго `<= 09:00`)
- Ориентиры: `main v8`: `CatBoost = 2.162`, `LGB = 2.273`; `main v10`: бленд `~2.12–2.13`; старт ветки `v11`: E2E `2.07287` (офлайн `BLEND = 2.06668`, `STACK = 2.06991`).
- **Офлайн A/B (`+h9 freq + h11 group_dev`, честная 3-fold CV):**
  - `OLD (Wave 1)`: `BLEND = 2.02840`, вложенный `STACK = 2.03091` (`cat_rmse = 2.20391` за `63.7s`).
  - **`NEW (Wave 2B)`:** `cat_top (top-150, d=7)` = **`2.14763`** (`33.6s`), `mlp_clnx3` = **`2.04500`**, `dual_hubx3` = **`2.04380`** -> **`BLEND = 2.01917`**, строгий вложенный **`STACK = 2.01976`** (`−0.01115 MAE`, зазор `0.00059`). На чистых `674` признаках без AutoResearch: **`BLEND = 2.01112`**, **`STACK = 2.01434`**.
- **Полный 35-минутный живой E2E-прогон `solution.py` (`hack_final_task2`, `Qwen3.8-27B`):**
  - **Итоговый OOF MAE: `2.02226`** (`choice = смесь`, веса: `mlpx8: 0.578`, `mlp_dualx7: 0.256`, `etx4: 0.065`, `cat_topx4: 0.056`, `cat_rmse: 0.045`) — **`−0.140 MAE`** к `main v8`, **`~−0.105 MAE`** к `main v10`, **`−0.0506 MAE`** к старту `v11`!
  - `SUBMISSION_OK` (`2219x2`, `min=5.033, med=20.322, max=50.543, std=9.265`), время `31:16`, пик RAM `3.80 GiB`, `3` вызова LLM (`53.1s`), `0` ошибок, `0` poison-дропов.

### 2. Task 1 (`hack_final_task1`, 322k строк, 2-fold хронологическая CV, PR-AUC)
- **Полный 35-минутный живой E2E-прогон `solution.py` (`hack_final_task1`, `Qwen3.8-27B`):**
  - **Итоговый OOF PR-AUC: `0.15374`** (пик `0.15393`, `choice = ранг-смесь`, веса: `catx3: 0.888`, `etx3: 0.075`, `mlp_dualx3: 0.037`).
  - `SUBMISSION_OK` (`77332x2`), время `31:20`, пик RAM `4.02 GiB`, `et` ускорен с `~309s` до `69s`, `mlp` со `152s` до `76s`, `0` ошибок.

### 3. Стресс-стенд Private LB (`tools/stress_private.py`) — **9 / 9 PASS**, `0` ошибок, `0` watchdog
- **Полный 8-кейсовый прогон (`--budget 60 --rows 2500 --large-rows 200000 --threads 4`):**
  - `fraud / PR-AUC`: `PASS` (Test PR-AUC `0.4357`, ROC-AUC `0.8477`, OOF `0.5796`, `0.31 GiB`, `31.1s`)
  - `fraud / ROC-AUC`: `PASS` (Test ROC-AUC `0.8460`, OOF `0.8814`, `0.31 GiB`, `29.0s`)
  - `multiclass / LogLoss` (12 классов + сдвиг долей): `PASS` (Test LogLoss **`0.3906`**, F1-macro `0.8601`, OOF **`0.3433`**, `mlp` соло `0.3581` против LGB `0.4131`, вес `0.639`, `0.31 GiB`, `43.7s`)
  - `multiclass / F1-macro`: `PASS` (Test F1-macro **`0.8630`**, OOF `0.8593`, `0.31 GiB`, `34.3s`)
  - `regression / RMSLE`: `PASS` (Test RMSLE **`0.9556`**, OOF `1.0468`, `0.30 GiB`, `38.9s`)
  - `regression / MAPE` (с точными нулями): `PASS` (`constant: false`, `zero_mape_hurdle`, `0.29 GiB`, `32.2s`)
  - `text / PR-AUC` (русские тексты + временной сплит): `PASS` (Test PR-AUC **`1.0000`**, `future_text_poison_delta = 0.0`, `1.05 GiB`, `33.8s`)
  - `large 200k / MAE`: `PASS` (Test MAE `0.8036`, `0.33 GiB`, `48.3s`)
- **Масштабный стресс-тест на 1 000 000 строк (`800k train + 200k test`, `--budget 180 --threads 8`):** **`PASS`** (Test MAE **`0.7972`**, `165.5s`, пик RAM **`1.13 GiB`**, `0` ошибок).

## Что отбраковано экспериментально
- `rm3 + sys_max` в матрице `Z_base` Nowcaster (ухудшали RMSE сенсоров против `lag1_anom + GCV`).
- Переменный `n_round` с ранней остановкой по блокам в LGBM-nowcaster (`217s` и рост дисперсии между блоками).
- Периодические Фурье-эмбеддинги PLR (`2.172` MAE против `2.030` у SwiGLU+Skip) и `max_cols = 55 / 120 / 180` (`2.091 / 2.054 / 2.075` против `2.030` на `80`).
- `CatBoost Huber` на Task 2 (`MAE = 4.39` — расходится; оставлены `MAE`, `RMSE` и `cat_top RMSE`).
- Переключение early stopping на Task 1 с `PRAUC`/`aucpr` на `logloss` (`0.1548 -> 0.1520`) и пост-сдвиг медианы на Task 2 (`+0.00023`).
- Добавление `torch` в `requirements.txt` (избыточно при нулевой стоимости и лучшем качестве `_NumpyTabularResNet`).

## Проверка регламента и релизный аудит
- Утечек будущего нет (`<= 09:00` / `< t`): **да** (`tools/release_check.py` `poison_db` на реальной БД `hack_final_task2`: `sensors: 0/526`, `events: 0/526`, `combined: 0/526`; `stress_private` `future_text_poison_delta = 0.0`).
- Хардкода под конкретные таблицы/колонки нет: **да**.
- Новые пакеты в `requirements.txt`: **нет** (ровно 14 пакетов из белого списка).
- Компиляция, тесты и плоский архив: **PASS** (`py_compile` OK; **`69 / 69 OK`** за `17.9s` в `tools/test_final.py`; `dist/solution_v11.zip` прошёл `audit_zip` — `14` плоских файлов, `133,094` байт, `0` CRLF, `0` секретов, `0` `sys.path`/`PYTHONPATH`, `SHA-256: 53cd74735f6e1d7d70d5f8d78c66b9dd79b65206606bbc8217b9c991385bd9c8`).
