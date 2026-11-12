# Predicting EV Driver Types from Charging Sessions

A multiclass machine learning project that predicts what **kind of driver** an electric-vehicle owner is — Commuter, Casual Driver or Long-Distance Traveller — from the behaviour recorded in a single charging session.

Six models were trained and compared end to end, from preprocessing pipeline to confusion matrix. **Gradient Boosting won with 79.2% accuracy and a 0.787 F1 score.**

## Why it matters

Charging networks and utilities have to decide *where* to build stations and *how much* power to reserve, before the drivers arrive. If a session's data alone can tell you which kind of driver produced it, you can:

- forecast demand per site and per hour,
- price dynamically for peak and off-peak use,
- place and size new charging stations more accurately.

## The data

| | |
| --- | --- |
| Source | Kaggle EV charging sessions dataset |
| Rows | 1,320 charging sessions |
| Features | 20 (10 numerical, 10 categorical) |
| Target | `User Type` — Commuter / Casual Driver / Long-Distance Traveller |

Features include battery capacity, charging duration, energy consumed (kWh), charging rate (kW), distance driven, state of charge, time of day, day of week, charger type and temperature.

## What the exploration showed

- **No single feature predicts the target on its own.** No numerical feature reached |r| > 0.3 against charging duration, so the signal had to come from combinations — which is exactly where tree-based models do well.
- **Long-Distance Travellers stand out:** longer sessions and higher energy consumption.
- **Time-of-day behaviour differs by driver type**, which is useful for load forecasting.
- **Mild class imbalance** (Commuter is the largest class), not severe enough to need SMOTE.

## Preprocessing

A single scikit-learn **`ColumnTransformer`** pipeline, so training and inference always apply the same steps:

| Feature type | Steps |
| --- | --- |
| Numerical | Mean imputation → standardization |
| Categorical | Mode imputation → one-hot encoding |

**Columns removed on purpose:** user ID, timestamps and location — they identify the row rather than describe behaviour, and keeping them risks **data leakage**, where the model "cheats" by memorizing identifiers instead of learning patterns.

**Split:** 60% train / 20% validation / 20% test.

## Models and results

| Model | Accuracy | F1 score | Key parameters |
| --- | --- | --- | --- |
| Logistic Regression | 0.758 | 0.751 | C = 0.1 |
| Decision Tree | 0.708 | 0.704 | max_depth = 10 |
| Random Forest | 0.788 | 0.781 | n_estimators = 100 |
| **Gradient Boosting** | **0.792** | **0.787** | early stopping |
| SVM (SVC) | 0.774 | 0.765 | C = 1.0 |
| Neural Network (MLP) | 0.770 | 0.760 | hidden layers (100, 50) |

Hyperparameters were chosen by grid search on the validation set; the test set was touched only once, at the end.

**Most predictive features** (from Random Forest importance): charging duration, state of charge, energy consumed.

## The most interesting result

The SVM confusion matrix shows persistent confusion between **Casual Drivers and Commuters** — and that is not a modelling failure, it is a finding. Their charging behaviour genuinely overlaps: a commuter who charges irregularly looks like a casual driver in the data. Separating them would need information this dataset does not contain, such as repeated visits to the same location over time.

## Tech stack

Python · pandas · NumPy · scikit-learn (`ColumnTransformer`, `Pipeline`, `GridSearchCV`) · Matplotlib · Seaborn · Jupyter

## Repository layout

```
notebooks/ev_user_classification.ipynb   full analysis: EDA, pipeline, 6 models, evaluation
docs/Final_Report.pdf                    written report with methodology and results
```

## What I would do next

- Cyclic encoding for time of day, so 23:00 and 01:00 are close rather than far apart
- SMOTE or class weights, to improve recall on the smallest class
- Ensemble voting or stacking across the top three models
- Add weather and geolocation features, which should help separate the overlapping classes

---
**Author:** Nikhil Chhatraband · IA651 Machine Learning final project, Master's in Data Science
