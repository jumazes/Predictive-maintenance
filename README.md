# Predictive Maintenance ML Project

**Topic:** Machine Learning Methods and Models for Predicting the Technical Condition of Industrial Equipment

Проект сделан под критерии защиты AI/ML-проекта: корректная постановка задачи, EDA, baseline, несколько моделей, imbalanced metrics, error analysis, explainability, работающий API, воспроизводимость и responsible AI.

## Что внутри

- `data/ai4i2020_offline_replica.csv` — готовый локальный dataset, 10 000 строк.
- `Predictive_Maintenance_Experiment_EXECUTED.ipynb` — уже выполненный notebook с outputs.
- `train.py` — воспроизводимое обучение и сохранение всех результатов.
- `models/xgboost_failure_risk.joblib` — готовый inference bundle.
- `app.py` — FastAPI prototype (`/health`, `/predict`, Swagger `/docs`).
- `results/` — метрики, ablation, false positives/negatives, error analysis.
- `figures/` — графики для презентации.
- `docs/DEFENSE_GUIDE_RU.md` — русский материал для защиты + теория + Q&A.
- `docs/TECHNICAL_REPORT_RU.md` — технический отчёт.
- `docs/ROLES_TEMPLATE.md` — таблица вклада участников.
- `presentation/` — итоговая презентация с фактическими результатами.

## Важная оговорка по данным

Официальный **UCI AI4I 2020 Predictive Maintenance Dataset** — synthetic benchmark на 10 000 наблюдений и 6 основных features. UCI указывает, что реальные predictive-maintenance datasets трудно публиковать, поэтому AI4I создан как synthetic dataset, отражающий промышленный сценарий.

В этом архиве находится `ai4i2020_offline_replica.csv`: **локальная воспроизводимая replica по опубликованной спецификации UCI**, а не byte-identical официальный CSV. Она нужна, чтобы проект запускался без интернета. Все цифры в приложенных results/presentation относятся именно к этой replica.

Для финальной сдачи можно скачать официальный CSV:

```bash
python download_official_uci.py
```

Источник: UCI Dataset 601, DOI `10.24432/C5HS5C`.

## Быстрый запуск

Python 3.11–3.13:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python train.py
python smoke_test.py
uvicorn app:app --reload
```

После запуска API:

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `GET /health`
- Prediction: `POST /predict`

Пример запроса:

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

## Эксперимент

Разбиение: 70% train / 15% validation / 15% locked test, stratified, `random_state=42`.

В predictive inputs входят только:

1. Type
2. Air temperature
3. Process temperature
4. Rotational speed
5. Torque
6. Tool wear

Дополнительно из них вычисляются три engineered features:

- temperature gap;
- mechanical power;
- wear × torque.

`UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` не используются как признаки. Failure-mode columns используются только **после prediction** для error analysis.

### Test results на bundled replica

| Model | Precision | Recall | F1 | F2 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.000 | 0.000 | 0.000 | 0.000 | 0.037 |
| Logistic Regression | 0.284 | 0.518 | 0.367 | 0.445 | 0.362 |
| Random Forest | 0.885 | 0.821 | **0.852** | **0.833** | 0.883 |
| XGBoost | 0.770 | **0.839** | 0.803 | 0.825 | **0.898** |

Для prototype выбран XGBoost из-за лучшего PR-AUC и recall. Threshold `0.375` выбран на validation set по F2-score, не на test set.

На locked test у XGBoost:

- TN = 1430
- FP = 14
- FN = 9
- TP = 47

Feature engineering увеличил XGBoost PR-AUC примерно с **0.798 до 0.895** в отдельном ablation experiment.

## Почему accuracy не главная метрика

В dataset только около 3.7% failure cases. Dummy model, которая всегда отвечает `Normal`, получает около 96.3% accuracy, но recall отказов = 0. Поэтому основной акцент: **Recall, F1/F2 и PR-AUC**.

## Responsible AI

Это educational prototype, а не система автоматической остановки оборудования. `risk_score` не называется calibrated probability. В реальном производстве необходимы human review, проверка на реальной telemetry, monitoring data drift, calibration, cost analysis ошибок и fail-safe business rules.

## Reproducibility

- random seed фиксирован;
- preprocessing и model объединены в sklearn Pipeline;
- зависимости закреплены в `requirements.txt`;
- готов `Dockerfile`;
- model bundle содержит threshold и metadata;
- `train.py` заново создаёт metrics/figures/model.
