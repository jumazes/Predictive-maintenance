# Технический отчёт
## Machine Learning Methods and Models for Predicting the Technical Condition of Industrial Equipment

## 1. Постановка задачи

Цель проекта — по текущим параметрам оборудования определить состояние, связанное с повышенным риском отказа. Основная AI-задача — supervised binary classification: `0 = normal`, `1 = failure`.

Практический пользователь — инженер по техническому обслуживанию. Модель не должна самостоятельно принимать решение об остановке оборудования; она создаёт дополнительный risk signal для проверки.

Критерий успеха определён не через accuracy, а через способность обнаруживать редкий failure class. Основные метрики: Recall, F1/F2 и PR-AUC.

## 2. Данные

Для воспроизводимого offline-запуска используется `ai4i2020_offline_replica.csv`, построенная по публично описанным правилам UCI AI4I 2020. Структура: 10 000 строк, 6 основных входных признаков, ID-поля, общий target `Machine failure` и пять failure-mode labels.

В текущей replica 373 failure cases (3.73%). Пропусков нет.

Predictive features:

- Type;
- Air temperature [K];
- Process temperature [K];
- Rotational speed [rpm];
- Torque [Nm];
- Tool wear [min].

Идентификаторы и failure-mode labels исключены из входов. Это предотвращает target leakage.

Разбиение выполнено stratified: 7000 train, 1500 validation, 1500 locked test. Random seed = 42.

## 3. Feature engineering

Из доступных sensor inputs вычисляются:

1. `Temperature gap = Process temperature − Air temperature`;
2. `Mechanical power = Torque × RPM × 2π/60`;
3. `Wear × torque`.

Они физически интерпретируемы и вычисляются из данных, доступных во время inference.

Ablation experiment показывает рост XGBoost PR-AUC с 0.798 на исходных шести features до 0.895 после добавления трёх engineered features.

## 4. Модели

Сравнены четыре подхода:

- DummyClassifier — baseline, показывающий ловушку accuracy;
- Logistic Regression — линейный baseline;
- Random Forest — bagging ensemble деревьев;
- XGBoost — gradient boosting деревьев.

Для Logistic Regression и tree ensembles учитывается дисбаланс классов. Для XGBoost используется `scale_pos_weight`. Decision threshold выбирается на validation set по F2-score: recall имеет больший вес, поскольку пропущенный отказ потенциально дороже лишней проверки.

## 5. Результаты

| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 0.963 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.037 |
| Logistic Regression | 0.933 | 0.284 | 0.518 | 0.367 | 0.445 | 0.902 | 0.362 |
| Random Forest | 0.989 | 0.885 | 0.821 | 0.852 | 0.833 | 0.975 | 0.883 |
| XGBoost | 0.985 | 0.770 | 0.839 | 0.803 | 0.825 | 0.987 | 0.898 |

XGBoost имеет лучший PR-AUC и highest recall среди сравниваемых ML-моделей. Random Forest показывает лучший thresholded F1/F2. Для prototype выбран XGBoost, поскольку он лучше ранжирует редкий positive class; threshold отдельно оптимизирован на validation set.

Confusion matrix XGBoost на test set при threshold 0.375:

- 1430 true negatives;
- 14 false positives;
- 9 false negatives;
- 47 true positives.

## 6. Error analysis

Failure-mode labels использованы только после prediction для понимания ошибок.

| Failure mode | Test cases | Detected | Recall |
|---|---:|---:|---:|
| TWF | 7 | 2 | 0.286 |
| HDF | 22 | 20 | 0.909 |
| PWF | 15 | 15 | 1.000 |
| OSF | 10 | 10 | 1.000 |
| RNF | 2 | 0 | 0.000 |

Модель уверенно обнаруживает deterministic/feature-related failure modes PWF и OSF и большинство HDF. TWF и особенно RNF существенно сложнее: часть таких событий в исходной логике AI4I задаётся случайно. Это важное ограничение: не все failures принципиально предсказуемы из текущих sensor features.

## 7. Explainability

Для XGBoost рассчитаны SHAP values. Наиболее важные признаки в данном эксперименте:

1. Rotational speed;
2. Tool wear;
3. Mechanical power;
4. Wear × torque;
5. Temperature gap;
6. Torque.

SHAP интерпретируется как объяснение поведения модели, а не доказательство причинно-следственной связи.

## 8. Prototype и engineering

Создан FastAPI prototype:

- `GET /health`;
- `POST /predict`;
- Swagger UI через `/docs`.

Model artifact содержит preprocessing Pipeline, XGBoost, decision threshold и metadata. Код запускается через `requirements.txt`; дополнительно предоставлен Dockerfile и smoke test.

API возвращает `risk_score`, threshold, predicted status и derived features. Термин `risk_score` выбран специально: class weighting означает, что score нельзя без calibration объявлять реальной вероятностью отказа.

## 9. Responsible AI и ограничения

- dataset synthetic; performance на реальном заводе не подтверждена;
- bundled CSV — offline replica UCI specification, а не официальный оригинал;
- возможны false negatives, поэтому модель не должна быть единственным источником решения;
- требуется human-in-the-loop;
- на реальных данных необходимы calibration, cost-sensitive thresholding, drift monitoring и регулярная переоценка;
- sensor quality, missing telemetry и изменение режима оборудования могут снизить performance.

## 10. Следующий этап

Для магистерской работы логичное развитие:

- перейти к real run-to-failure/time-series telemetry;
- добавить Remaining Useful Life regression;
- сравнить classical ML с sequence models;
- внедрить experiment tracking/model registry/drift monitoring;
- проверить экономический cost false positives и false negatives.
