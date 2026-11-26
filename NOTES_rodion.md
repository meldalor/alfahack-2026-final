# Результаты ветки `exp/rodion-nowcast-ensemble-leakfix` (Родион)

- Ветка: `exp/rodion-nowcast-ensemble-leakfix` (создана от `origin/main` `6b499e3`)
- Затронутые файлы: `ag_database.py`, `ag_relational.py`, `ag_nowcast.py` (новый), `ag_features.py`, `ag_research.py`, `ag_metrics.py`, `ag_models.py`, `ag_ensemble.py`, `solution.py`, `tools/make_zip.py`, `tools/test_final.py`, `NOTES_rodion.md`

## Что проверено
- **Аудит утечек `main` (`6b499e3`) и закрытие каналов утечки в AutoResearch / признаках на временных сплитах:**
  - На `hack_final_task2` подтверждено, что в `main` базовый `_temporal_join` и `causal_stream_features` чистые, но слой AutoResearch и `prepare()` допускали утечки из будущих строк `test`: `op_freq(value_freq_cols)` менял `37/40` колонок при удалении будущих строк `test`, `op_group_dev` — `2/10`, высококардинальные категории (`>1000`) в `prepare()` использовали `train+test`, а промпт в `ag_research.py` рекомендовал статистики по `concat(train, test)`. Все эти каналы переведены в режим строго `train_only` / отключены на временных сплитах.
  - Расширен фильтр постфактум-колонок в `_interval_join` (`actual|final|realized|outcome|result|resolved`), добавлена защита интервальных таблиц `(start_*, end_*)` от попадания в статический `many-to-one` (кроме таблиц сущностей со своим первичным ключом вида `<singular_table>_id`, таких как `campaigns` по `campaign_id`).
- **Реляционные признаки, потоковое чтение БД и строго каузальный 1-step-ahead Nowcaster (`ag_nowcast.py`):**
  - В `ag_database.py` `io.BytesIO()` заменён на `tempfile.SpooledTemporaryFile(max_size=64MB)` для снижения пика RAM при `COPY`.
  - В `_temporal_join` добавлены кросс-групповые общесистемные агрегаты на тот же завершённый интервал (`__sys_mean`, `__sys_dev`), исправлен детектор бинарности `__runlen` по первой временной половине (`b[key] <= b[key].median()`), добавлен экспорт регулярных панелей в `meta['panels']`.
  - Добавлен модуль `ag_nowcast.py`: expanding-window 1-step-ahead прогноз состояния регулярных панелей (`сезонный профиль + многовыходный Cholesky Ridge + топ-6 LightGBM`), выровненный по границам `plan.folds` и `test_cutoff`, без какого-либо доступа к текущему или будущему слоту (`t_obs <= slot - cadence`).
  - В `ag_features.py` добавлены: `origin` для `__ord` строго от `min(train)` (сохраняет секундную точность в `float32` и исключает зависимость от `test`) + точные разности пар дат (`__minus__`); защита `{0, 1}` флагов от перевода в категории по `cat_hint`; быстрый кешированный кросс-табличный матчинг списков тегов через запятую (`__n_tags`, `__ov__`, `__jac__`, словарь строго по `train`); строго-прошлые (`< t`) групповые потоковые признаки `stream_grp__*` и фикс граничного условия `pos_in_cur_hr` (`lp >= j_hr`); фильтрация присоединённых числовых телеметрических колонок из `entity_key_candidates` и `_DERIVED_PFX` из `cross_impute_outages`/`entity_lag_cols`.
- **Модели, функции потерь, неотрицательный симплекс-ансамбль и вложенный стекинг:**
  - В `ag_metrics.py` добавлена пофолдовая `oof_score()` для ранжирующих метрик (`roc_auc`, `average_precision`).
  - В `ag_models.py` добавлены: `lgb_objective` с оценкой `huber`/`fair`/`l2` по `l1` на MAE-задачах; `fit_xgb` с `objective_override='pseudohuber'` (`huber_slope=0.5*std(yt)`) и калиброванным `base_score`; `fit_hgb` (`HistGradientBoosting` с нативным `absolute_error` и категориями); `fit_cat` с `loss_override`, `l2_leaf_reg` и `border_count=128` на широких не-мультикласс задачах (`>200` признаков); исправление `fit_linear` и `fit_mlp` на временной регрессии (исключение монотонных `__ord`/`__year`, отбор `top_cols`, `QuantileTransformer(output_distribution='normal')` для MLP).
  - В `ag_ensemble.py` реализована оптимизация весов на неотрицательном симплексе `_simplex_reg_weights` (`w >= 0, sum(w) = 1`) напрямую под целевую метрику регрессии как в `blend()`, так и во вложенном OOF-стекинге `stack_blend()` по границам `plan.folds` (полностью устранены отрицательные веса вроде `-0.19` / `-0.128` от `HuberRegressor`).
  - В `solution.py` подключены `build_nowcast_features`, мульти-loss модели (`lgb_huber`, `cat_rmse`, `xgb_ph`), `hgb`, передача `top_cols` в `fit_linear`/`fit_hgb`/`fit_mlp` и `plan` в `blend`/`rank_blend`/`stack_blend`/`pool_results`.
- **Юнит-тесты, сборка ZIP и серверная проверка в `/root/alfahack/exp_rodion`:**
  - В `tools/test_final.py` добавлено 6 новых тестов (итого `37/37 PASSED` локально и на сервере).
  - Двойной poison-аудит и онлайн-стрим аудит на реальной БД `hack_final_task2` (пробная строка `452`, `2025-08-01 06:50:00`).
  - Полные E2E-прогоны `solution.py` с реальным LLM на `hack_final_task1` (`480s`) и `hack_final_task2` (`900s`), плюс прямой 3-fold 9-модельный бенчмарк на `hack_final_task2`.

## Что сработало (с цифрами против main)
- **Task 2 (`hack_final_task2`, 3-fold хронологическая CV по `timestamp__ord`, MAE, строго `<= 09:00`):**
  - Ориентиры `main` (`6b499e3`): `LightGBM = 2.273`, `XGBoost = 2.251`, `CatBoost (depth=6) = 2.162` (с AutoResearch); в прямом 3-fold бенчмарке без AutoResearch у `main`: `cat_d6 = 2.20159`, бленд `2.17265`, `mlp ~ 3.09`.
  - Прямой 3-fold бенчмарк нашей ветки (без AutoResearch, 623 признака, `0/648` утечек):
    - `lgb_l1`: `2.28708` (`87.6s`)
    - `lgb_huber`: **`2.20691`** (`30.1s`, **−0.066 MAE** к `main` LightGBM `2.273`, в 2.9× быстрее `lgb_l1`)
    - `xgb_l1`: `2.24914` (`190.5s`) -> `xgb_ph` (PseudoHuber): **`2.24195`** (`85.8s`, в 2.2× быстрее)
    - `hgb` (L1): **`2.24279`** (`39.1s`)
    - `mlp` (`QuantileTransformer`, без `__ord`/`__year`, `top_cols`): **`2.37586`** (`28.8s`, вместо `~3.09` в `main`)
    - `cat_d6` (`border_count=128`): `2.19482` (`101.9s`, ускорение со `176s` до `102s`)
    - `cat_d6_rmse` (`loss_override='RMSE'`): **`2.18734`** (`87.1s`)
    - `cat_d7` (`border_count=128`): **`2.17331`** (`132.3s`)
    - **Симплекс-ансамбль (`blend`):** **`2.12180`** (`cat_d7: 0.333`, `mlp: 0.218`, `cat_d6_rmse: 0.169`, `lgb_huber: 0.135`, `xgb_ph: 0.107`, `cat_d6: 0.039`, все веса `>= 0`) — улучшение **−0.051 MAE** против бленда `main` (`2.17265`) и **−0.040 MAE** против `CatBoost 2.162` из `main`.
    - **Вложенный OOF-стекинг (`stack_blend`, строго по 3 временным фолдам):** **`2.13577`** (все веса `>= 0`).
  - Полный E2E-прогон `solution.py` с LLM (`AGENT_TIME_BUDGET=900`): AutoResearch принял `#7 Huber (+0.08848, 3/3 фолдов)` и `#12 pairwise (+0.02421, 2/3 фолдов)` (`2.31246 -> 2.19977`); `lgb_main (Huber) = 2.19465` (**−0.078 MAE** к `main` `2.273`), `xgb_ph = 2.23220`, `hgb = 2.25020`, итоговый бленд (на 900s бюджете без CatBoost до правки `border_count=128`) = **`2.16950`**, `submission.csv` `(2219, 2), nan=0`.
- **Task 1 (`hack_final_task1`, 2-fold хронологическая CV, PR-AUC, 322k / 77k):**
  - Ориентир `main` (`6b499e3`): `0.1535–0.1552`.
  - Полный E2E-прогон `solution.py` с LLM (`AGENT_TIME_BUDGET=480`): **`0.15449`** (`ранг-смесь`: `xgb 0.667 + hgb 0.333`; `xgb=0.15353`, `hgb=0.15326`, `lgb_fast=0.15288`, `lgb_main=0.15206`, `logreg=0.14883`), `submission.csv` `(77332, 2), nan=0`.
  - Подтверждён джойн `campaigns` (`many-to-one on ['campaign_id'], cov=1.000, +5 feats`) и автодетект кросс-табличного тег-матчинга (`ads__target_interests__ov__users__interests`, `__jac__`): на 400k строках `extract=2.2s, enrich=1.9s, stream=2.6s, prepare=30.2s`.
- **Время / пик RAM (сервер 8 CPU, 32 ГБ RAM):**
  - Task 2 E2E (`900s`): `835.6s`, пик RAM **`3.49 GB`** (`3659176 KB`).
  - Task 2 9-модельный бенчмарк (`bench_cat.py`, Nowcast `67.9s` + 3×CatBoost): `905.7s`, пик RAM **`3.38 GB`** (`3545636 KB`).
  - Task 1 E2E (`480s`, 400k строк): `409.7s`, пик RAM **`3.02 GB`** (`3169484 KB`).

## Что не сработало
- Условное включение правила завершённого интервала по регулярному выражению из `task.md` оказалось ненадёжным на коротких формулировках задачи — оставлено безусловное правило `rt + cadence <= event_time` из `main`.
- В первичном варианте `ag_relational.py` проверка бинарности для `__runlen` через `iloc[:len//2]` на таблице, отсортированной по `(district_id, timestamp)`, затрагивала хвост 2025 года и ломалась при отравлении будущих строк; исправлено на `b[key] <= b[key].median()`.
- В первичном варианте `ag_nowcast.py` порог `(te_pidx >= 0).mean() < 0.5` мог отбрасывать панель при появлении новых групп в `test`, а многопоточный BLAS (8 потоков) вносил 1-ULP шум float32 (`1.22e-4`) при смене размера батча; заменено на `not (te_pidx >= 0).any()`, окно Ridge ограничено `150k` последними строками с `threadpool_limits(min(4, n_jobs))` -> аудит дал `0 / 648`.
- `fit_cat` на 623 признаках с дефолтным `border_count=254` требовал `~176s` на первый фолд и отменялся по тайм-лимиту на коротком 900s прогоне; переход на `border_count=128` при `>200` признаках ускорил `cat_d6` до `101.9s` суммарно на 3 фолда + refit без потери качества.
- Автоматический перехват всех таблиц с `(start_*, end_*)` в `_interval_join` первоначально перехватывал `campaigns` в Задаче 1; исправлено проверкой наличия собственного уникального ключа `<singular_table>_id` (`campaigns` вернулся к `many-to-one, cov=1.000`).

## Где смотреть код
- `ag_database.py`: `_PsycopgDatabase.table` (строки 38–53), `_LibpqDatabase.table` (строки 109–136) — `SpooledTemporaryFile` вместо `BytesIO`.
- `ag_relational.py`: `POST_FACTUM_RE` (строка 9); `_temporal_join` (строки 88–217: экспорт панелей 130–146, `__sys_mean`/`__sys_dev` 165–169, фикс `__runlen` 170–181); `_interval_join` (строки 220–264); `enrich` (строки 368–488: защита `(start_*, end_*)` с исключением для собственного PK 382–391, заполнение `meta` 476–487).
- `ag_nowcast.py`: целиком новый модуль (строки 1–413) — `_merge_panel_group`, `_build_causal_design`, `_seasonal_profile`, `_select_top_targets`, `build_nowcast_features`.
- `ag_features.py`: `_expand_datetime` (строки 43–75: `origin` по `train`); `prepare` (строки 113–127: защита `{0, 1}` флагов; 183–196: точные `__minus__` разности дат; 198–250: кросс-табличный тег-матчинг `__n_tags`/`__ov__`/`__jac__`; 252–271: `train`-only высококардинальные категории `>1000`); `op_freq`/`op_dup_count`/`op_group_dev` с `train_only` (строки 332–390); `entity_lag_cols`/`entity_key_candidates`/`cross_impute_outages` (строки 446–533, 742–750); `op_text_oof` с `train`-only TF-IDF при `periods is not None` (строки 600–673); `causal_stream_features` (строки 898–993: `bin_cols` по `train`, фикс `pos_in_cur_hr`, групповые `stream_grp__*`).
- `ag_research.py`: `seed_queue` (строки 240–276: `Huber` для MAE, отключение `value_freq_cols` и `dup_count` на временных сплитах); `materialize` (строки 328–372: `train_only=True` для `freq`/`group_dev`, поддержка `l2`); `propose` и `normalize` (строки 744–825: анти-утечный промпт для временных сплитов).
- `ag_metrics.py`: `RANK_METRICS` и `oof_score` (строки 30, 200–216).
- `ag_models.py`: `lgb_objective` (строки 79–95); `refit_rounds` (строки 300–310); `fit_cat` (строки 333–423: `loss_override`, `l2_leaf_reg`, `border_count=128`); `fit_xgb` (строки 429–525: `pseudohuber`, `base_score`); `fit_linear` (строки 597–700); `fit_hgb` (строки 706–803); `fit_mlp` (строки 806–896).
- `ag_ensemble.py`: `_eval`, `_simplex_reg_weights`, `blend`, `rank_blend`, `stack_blend` (строки 18–235).
- `solution.py`: `run`/`solve_target` (строки 232–243, 308–371: `rel_meta` + `build_nowcast_features`); `model_family`/`pool_results` (строки 439–463); `final_stage` (строки 691–750, 812–833: `lgb_huber`, `cat_rmse`, `xgb_ph`, `hgb`, `top_cols`, `lgb_huber_seed`).
- `tools/test_final.py`: 6 новых юнит-тестов (строки 433–568).

## Проверка регламента
- Утечек будущего нет (`<= 09:00` / `< t`): **да** (проверено полным двойным poison-аудитом на `hack_final_task2` для строки `452`: отравление всех 5 почасовых таблиц `>= 06:00` + отравление всех будущих/одновременных строк `test` -> **`0 / 648`** изменённых колонок; онлайн-стрим с полным удалением будущих строк `test` -> **`0 / 648`** изменённых колонок; включая `Nowcaster`, `__sys_mean`/`__sys_dev`, `__runlen`, `stream__*`, `stream_grp__*`, `op_freq(train_only=True)`, `op_group_dev(train_only=True)`).
- Хардкода под конкретные таблицы/колонки нет: **да** (все механизмы — панельный Nowcaster, кросс-групповые агрегаты, тег-матчинг, групповые потоки, мульти-loss модели, симплекс-стекинг — определяются автоматически по схеме и типам данных).
- Новые пакеты в `requirements.txt`: **нет** (ровно 14 пакетов из белого списка регламента без изменений, `0s` дополнительных накладных расходов).
- `tools/test_final.py`: **passed** (`37/37 PASSED` локально в WSL и на сервере `/root/alfahack/exp_rodion`; плоский ZIP без CRLF, без секретов и без `sys.path`/`PYTHONPATH`).
