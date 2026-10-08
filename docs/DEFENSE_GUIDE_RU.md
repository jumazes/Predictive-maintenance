# Защита Predictive Maintenance — русский handbook

## 0. Главная логика проекта в одной фразе

**Мы не пытаемся “угадать любую поломку”. Мы проверяем, насколько по шести эксплуатационным параметрам можно обнаружить состояние с повышенным риском failure, корректно сравниваем несколько ML-моделей на imbalanced data, анализируем ошибки и показываем работающий API.**

Запомнить цепочку:

`Problem → Data → Leakage control → Baseline → RF/XGBoost → Validation threshold → Locked test → Error analysis → SHAP → API → Limitations`

---

# 1. План защиты на 14 минут

## Презентация — 6 минут

### 0:00–0:30 — Слайд 1. Тема

> Тема проекта — прогнозирование технического состояния промышленного оборудования методами машинного обучения. В текущей версии задача сформулирована как binary classification: по sensor и operating parameters определить normal state или состояние, связанное с failure.

### 0:30–1:00 — Слайд 2. Проблема и актуальность

> Reactive maintenance происходит уже после поломки, а preventive maintenance работает по фиксированному расписанию. Predictive maintenance использует фактическое состояние оборудования. Для инженера полезен ранний risk signal, который помогает решить, какое оборудование проверить в первую очередь. Поэтому AI здесь нужен для поиска нелинейных зависимостей между температурой, скоростью, torque, wear и failure state.

### 1:00–1:25 — Слайд 3. Что уже сделано / Казахстан

> Сама идея predictive maintenance не новая, и мы это не скрываем. Есть исследования на нефтяных скважинах Казахстана, clustering эксплуатационных режимов и работы по Remaining Useful Life подшипников. Поэтому наша цель не “изобрести predictive maintenance”, а построить воспроизводимый baseline pipeline, корректно сравнить модели, провести error analysis и сделать работающий prototype.

### 1:25–2:05 — Слайды 4–5. Данные и EDA

> В эксперименте 10 тысяч наблюдений и шесть основных inputs: type, air/process temperature, RPM, torque и tool wear. Failure class редкий — около 3.7%. Поэтому accuracy опасна: Dummy baseline, который всегда говорит Normal, получает 96.3% accuracy, но recall failure равен нулю.
>
> ID и TWF/HDF/PWF/OSF/RNF мы не используем как predictive features. Эти колонки непосредственно связаны с target и создали бы leakage. Split — 70/15/15 stratified: train, validation и locked test.

### 2:05–2:50 — Слайды 6–7. Метод

> Baseline — Logistic Regression. Затем Random Forest и XGBoost. Мы добавили три engineered features: temperature gap, mechanical power и wear × torque. Они вычисляются только из доступных inputs, поэтому leakage нет.
>
> Imbalance учитываем через class weights / scale_pos_weight. Threshold выбираем только на validation по F2, потому что пропущенный failure для этой задачи важнее лишней проверки. Test не используется для подбора threshold.

### 2:50–3:50 — Слайд 8. Результаты

> Dummy подтверждает, что accuracy вводит в заблуждение. Logistic Regression даёт PR-AUC 0.362. Random Forest достигает F1 около 0.852 и PR-AUC 0.883. XGBoost показывает лучший PR-AUC — 0.898 — и recall 0.839.
>
> Для prototype мы выбрали XGBoost как лучший по ranking quality редкого positive class. На locked test он обнаружил 47 из 56 failures: 47 TP, 9 FN, при этом 14 false alarms.

### 3:50–4:35 — Слайды 9–10. Эксперименты и error analysis

> Feature engineering дал измеримое улучшение: PR-AUC XGBoost вырос примерно с 0.798 до 0.895 в ablation experiment.
>
> Затем мы разобрали ошибки по hidden failure modes, которые не использовались при обучении. PWF и OSF обнаружены полностью, HDF — примерно на 91%. TWF и RNF заметно хуже. Это логично: часть таких failures в AI4I задаётся случайно и не полностью определяется нашими sensor inputs. Это показывает реальное ограничение модели, а не просто “плохую метрику”.

### 4:35–5:15 — Слайд 11. Explainability и demo architecture

> SHAP показывает, что наиболее важны RPM, tool wear, mechanical power, wear × torque и temperature gap. SHAP объясняет поведение модели, но не доказывает причинность.
>
> Модель сохранена вместе с preprocessing и threshold. FastAPI принимает шесть параметров, рассчитывает derived features и возвращает status и risk score.

### 5:15–6:00 — Слайд 12. Ограничения и вывод

> Мы сознательно не называем risk score точной вероятностью отказа: class weighting нарушает calibration. Модель также не должна автоматически останавливать оборудование — нужен инженер и human-in-the-loop. Основное ограничение — synthetic data; для промышленного использования нужна проверка на реальной telemetry.
>
> Итог: мы построили полностью воспроизводимый ML pipeline, показали improvement над baseline, провели error analysis и сделали работающий prototype. Следующий этап — real time-series data и Remaining Useful Life.

---

# 2. Демонстрация — 5 минут

Не трать время на запуск notebook с нуля. Всё должно быть заранее установлено, но показывай систему live.

## 0:00–0:40 — структура проекта

Открыть репозиторий и быстро показать:

- `Predictive_Maintenance_Experiment_EXECUTED.ipynb`
- `train.py`
- `results/`
- `models/`
- `app.py`
- `README.md`
- `Dockerfile`

Фраза:

> Notebook нужен для исследования и объяснения эксперимента, `train.py` — для воспроизводимого обучения, а API использует уже сохранённый pipeline.

## 0:40–1:40 — notebook

Показать три места:

1. split и leakage control;
2. таблицу metrics;
3. PR curve / confusion matrix / error analysis.

Не листать всё подряд.

## 1:40–2:10 — запуск API

```bash
uvicorn app:app --reload
```

Открыть `/docs`.

## 2:10–3:20 — Normal example

В Swagger вызвать `/predict` с обычным режимом:

```json
{
  "type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.2,
  "rotational_speed_rpm": 1550,
  "torque_nm": 38.0,
  "tool_wear_min": 80
}
```

Объяснить output: `risk_score`, `decision_threshold`, derived features.

## 3:20–4:20 — Risk example

```json
{
  "type": "L",
  "air_temperature_k": 300.0,
  "process_temperature_k": 307.8,
  "rotational_speed_rpm": 1320,
  "torque_nm": 58.0,
  "tool_wear_min": 190
}
```

В bundled model этот пример даёт высокий risk score и `FAILURE_RISK`.

Фраза:

> API не говорит “станок точно сломается”. Он говорит, что текущий набор параметров попадает в high-risk region модели и требует проверки.

## 4:20–5:00 — reproducibility

Показать `README`, `requirements.txt`, `Dockerfile`, `smoke_test.py`.

> Все версии зависимостей зафиксированы, preprocessing хранится внутри Pipeline, threshold сохранён вместе с моделью, поэтому inference не расходится с training.

---

# 3. Теория, которую нужно реально понимать

## Binary classification

Два класса: `0 Normal`, `1 Failure`.

Модель сначала выдаёт score. Затем:

`score >= threshold → Failure`

`score < threshold → Normal`

Threshold — бизнес/инженерное решение, а не обязательные 0.5.

## Confusion matrix

- **TP** — failure был и модель его нашла.
- **TN** — normal и модель сказала normal.
- **FP** — false alarm: normal помечен как risk.
- **FN** — failure пропущен. Для maintenance это обычно наиболее опасная ошибка.

Наш XGBoost test: `TP=47, FN=9, FP=14, TN=1430`.

## Precision

`Precision = TP / (TP + FP)`

Из всех alerts сколько реально являются failure cases.

Высокий precision = меньше лишних проверок.

## Recall

`Recall = TP / (TP + FN)`

Из всех реальных failures сколько модель нашла.

Для maintenance recall особенно важен, потому что FN означает пропущенный риск.

## F1 и F2

F1 — harmonic mean precision и recall.

F2 сильнее весит recall. Мы используем F2 на validation при выборе threshold, потому что для нашего сценария FN считаем более критичным.

## ROC-AUC vs PR-AUC

ROC-AUC измеряет ranking между positive и negative classes, но при сильном imbalance может выглядеть очень хорошо из-за огромного количества negatives.

PR-AUC фокусируется на precision/recall positive class и поэтому информативнее для редких failures.

В нашем проекте baseline prevalence ≈ 0.037, а XGBoost PR-AUC ≈ 0.898.

## Class imbalance

Failures ≈ 3.7%. Поэтому модель может “выиграть” по accuracy, почти всегда отвечая Normal.

Меры:

- stratified split;
- class weights / `scale_pos_weight`;
- failure-focused metrics;
- threshold tuning.

## Train / validation / test

- Train — учим параметры модели.
- Validation — выбираем threshold и решения дизайна.
- Test — один раз проверяем финальное качество.

Если выбирать threshold по test, test перестаёт быть независимым и результат становится optimistic.

## Data leakage

Leakage — модель получает информацию, которая в реальном inference недоступна или напрямую раскрывает target.

У нас `TWF/HDF/PWF/OSF/RNF` нельзя подавать как features: они описывают причины failure. Мы используем их только после prediction для error analysis.

## Logistic Regression

Линейная модель. Хороша как baseline: быстрая, понятная, показывает, хватает ли простой линейной зависимости.

## Random Forest

Много decision trees обучаются на разных bootstrap samples/features; их ответы агрегируются. Устойчив к нелинейным зависимостям и хорошо работает с tabular data.

## XGBoost

Gradient boosting: деревья строятся последовательно, каждое новое дерево исправляет ошибки предыдущего ensemble. Хорош для сложных взаимодействий табличных признаков.

## Feature engineering

Мы не просто “дали модели больше колонок”. Derived features содержат физически осмысленные взаимодействия: power связывает torque и RPM, wear×torque связывает нагрузку и накопленный wear.

Ablation показывает, что это реально улучшило PR-AUC.

## SHAP

SHAP распределяет вклад признаков в output модели на основе идеи Shapley values.

Важно:

- SHAP объясняет model prediction;
- SHAP ≠ causal effect;
- высокий SHAP feature не означает, что изменение этого sensor обязательно физически вызовет failure.

## Calibration

Model score 0.9 не всегда означает “90% реальная вероятность”. Class weighting особенно может сместить calibration.

Поэтому API пишет `risk_score`, а не `failure_probability`.

## Overfitting

Модель запоминает training data и плохо работает на новых данных. Контроль: separate test, regularization, ограничение дерева, сравнение train/validation, reproducible split.

---

# 4. Вопросы комиссии и ответы

## Почему вообще ML, если failure rules в AI4I известны?

> В реальном сценарии инженер видит sensor values, но не скрытую failure-mode label. Мы моделируем именно этот inference: из доступных sensor features получить risk signal. Кроме того, проект оценивает весь pipeline — imbalance, model comparison, explainability и deployment. При этом мы честно используем известные failure modes только после prediction для error analysis.

## Почему не accuracy?

> Failure class редкий. Dummy получает 96.3% accuracy и при этом не находит ни одного failure. Поэтому accuracy не отражает цель проекта.

## Почему PR-AUC?

> Он оценивает precision–recall trade-off именно для редкого positive class и гораздо информативнее при сильном imbalance.

## Почему XGBoost, если Random Forest имеет выше F1?

> Random Forest действительно лучше по thresholded F1/F2 на этом конкретном test split. XGBoost лучше по PR-AUC и recall, то есть лучше ранжирует failure cases. Для prototype мы выбрали XGBoost, а это trade-off, а не утверждение, что одна модель лучше по каждой метрике.

## Почему threshold 0.375?

> Он выбран на validation set по максимальному F2. Test set при выборе threshold не использовался.

## Почему F2?

> F2 сильнее штрафует низкий recall. В maintenance пропустить failure обычно опаснее, чем отправить оборудование на лишнюю проверку.

## Почему не SMOTE?

> Мы начали с class weighting, потому что он не создаёт synthetic minority rows и проще контролируется. SMOTE можно включить как отдельный experiment, но он должен применяться только внутри training folds, иначе появится leakage.

## Почему engineered features — это не leakage?

> Они вычисляются только из шести доступных input sensors. Мы не используем target и hidden failure-mode columns.

## Почему TWF/RNF плохо определяются?

> В AI4I часть этих механизмов содержит случайность. Если событие не определяется наблюдаемыми inputs, ML не может восстановить отсутствующую информацию. Это фундаментальное ограничение feature set.

## Что показывает SHAP?

> Вклад feature в output конкретной обученной модели. Он помогает понять логику model behaviour, но не доказывает физическую причинность.

## Что означает risk score 0.9?

> Это model score. Мы не утверждаем, что это 90% физическая вероятность failure, потому что class weighting и отсутствие calibration не позволяют делать такую интерпретацию.

## Можно ли реально поставить эту систему на завод?

> Нет, не в текущем виде. Сначала нужно обучить и валидировать её на real telemetry конкретного оборудования, провести calibration, economic cost analysis, safety review и организовать drift monitoring.

## В чём вклад проекта, если predictive maintenance уже существует?

> Новизна текущего coursework не в новом ML-алгоритме. Наш вклад — корректный reproducible baseline: leakage-safe preprocessing, imbalance-aware evaluation, physically motivated feature engineering с ablation, failure-mode error analysis, explainability и работающий inference API. Для магистерской research novelty нужно усиливать через real time-series/RUL/industrial data.

## Почему dataset synthetic?

> UCI прямо объясняет, что реальные predictive-maintenance data трудно публиковать. Synthetic benchmark позволяет сделать воспроизводимый эксперимент. Это одновременно limitation, поэтому мы не переносим test scores на реальные предприятия.

## У вас официальный UCI CSV?

> В приложенном offline-комплекте используется reproducible AI4I-style replica по опубликованной спецификации, чтобы проект запускался без сети. Она явно маркирована, и все показанные результаты относятся к ней. В репозитории есть отдельный downloader для official UCI dataset; при финальной industrial validation результаты необходимо пересчитать на официальных или реальных данных.

---

# 5. Что проверить за 15 минут до защиты

1. `python smoke_test.py` проходит.
2. `uvicorn app:app --reload` запускается.
3. `/docs` открывается.
4. Оба demo JSON сохранены в отдельном txt/буфере.
5. Notebook уже выполнен — не запускай training во время защиты.
6. Презентация открывается без сломанных шрифтов/графиков.
7. README виден.
8. Таблица вкладов заполнена реальными именами.
9. Ты можешь без подсказки объяснить: PR-AUC, recall, leakage, threshold, SHAP, class imbalance.

# 6. Главная позиция на защите

Не говори: **“модель предсказывает поломку с 98.5% точностью”**. Это будет методологически слабая фраза.

Говори:

> “На locked test XGBoost имеет recall 83.9% и PR-AUC 0.898. Accuracy 98.5% мы показываем только как дополнительную метрику, потому что dataset сильно imbalanced.”

И не говори: **“0.8 = 80% вероятность поломки”**.

Говори:

> “Это risk score; для вероятностной интерпретации нужна отдельная calibration.”
