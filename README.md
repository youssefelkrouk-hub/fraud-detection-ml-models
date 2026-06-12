# Credit Card Fraud Detection — Project Summary

> **Notebook:** `credit_card_fraud_detection_FR.ipynb`  
> **Models:** `RandomForestClassifier`, `AdaBoostClassifier`, `CatBoostClassifier`, `XGBoost`, `LightGBM`  
> **Primary Metric:** ROC-AUC

---

## 1. Problem Description and Data

### Context

The dataset contains credit card transactions made by European cardholders in **September 2013**, over a period of two consecutive days.

| Attribute | Value |
|---|---|
| Total number of transactions | 284,807 |
| Number of fraudulent transactions | 492 |
| Fraud ratio | **0.172%** |

The dataset is therefore **extremely imbalanced**: the positive class (fraud, `Class = 1`) accounts for less than 0.2% of all observations.

### Data Structure

All input features are numerical, derived from a **PCA** transformation (the original data is confidential):

- **V1 to V28**: principal components (PCA);
- **Time**: seconds elapsed since the first transaction;
- **Amount**: transaction amount (usable for cost-sensitive learning);
- **Class**: binary target variable (`0` = legitimate, `1` = fraud).

No missing values were detected in the dataset.

---

## 2. Exploratory Data Analysis

### 2.1 Temporal Analysis

The temporal distribution of fraudulent transactions is **more uniform** than that of legitimate transactions. Fraudulent activity also occurs at night (off-peak hours in Europe), unlike legitimate transactions which follow a daytime cycle.

Hourly visualizations show that:
- The **total amount** and **number of legitimate transactions** fluctuate significantly by hour;
- **Fraudulent transactions** remain stable over time.

### 2.2 Amount Distribution

- **Legitimate** transactions have a higher mean, a larger Q1, and significant outliers;
- **Fraudulent** transactions have a lower mean, a higher Q4, and smaller outliers.

### 2.3 Correlations (Pearson)

Consistent with the PCA construction, features **V1–V28** are not correlated with each other. However, notable correlations exist with `Amount`:

| Type | Pairs |
|---|---|
| Positive correlation | `V7` ↔ `Amount`, `V20` ↔ `Amount` |
| Negative correlation | `V1` ↔ `Amount`, `V5` ↔ `Amount` |
| Negative correlation (time) | `V3` ↔ `Time` |

### 2.4 Feature Distributions (KDE Plots)

KDE distributions reveal the discriminative power of each feature:

| Separability | Features |
|---|---|
| Strong | **V4**, **V11** |
| Partial | **V12**, **V14**, **V18** |
| Distinct | **V1**, **V2**, **V3**, **V10** |
| Weak | **V25**, **V26**, **V28** |

PCA features for `Class = 0` are generally centered around 0. Fraudulent transactions (`Class = 1`) exhibit asymmetric distributions.

---

## 3. Predictive Models

### Train/Validation/Test Split Strategy

The dataset is divided into three sets using `train_test_split`:

```
Train      : ~60%
Validation : ~20%
Test       : ~20%
```

### 3.1 RandomForestClassifier

**Parameters:** `gini` criterion, 100 trees (`n_estimators=100`), 4 parallel jobs.

- **Feature importance**: the most important features are **V17**, **V12**, **V14**, **V10**, **V11**.
- The confusion matrix shows good fraud detection with some false positives.
- **ROC-AUC ≈ 0.85**

### 3.2 AdaBoostClassifier

**Parameters:** `SAMME.R` algorithm, `learning_rate=0.8`, 100 estimators.

- Lower performance than RandomForest on this imbalanced dataset.
- **ROC-AUC ≈ 0.83**

### 3.3 CatBoostClassifier

**Parameters:** 500 iterations, `learning_rate=0.02`, `depth=12`, `eval_metric='AUC'`.

- Slightly outperforms AdaBoost thanks to optimized gradient boosting.
- **ROC-AUC ≈ 0.86**

### 3.4 XGBoost

**Parameters:** `objective='binary:logistic'`, `eta=0.039`, `max_depth=2`, `subsample=0.8`, `eval_metric='auc'`, early stopping at 50 rounds.

- Uses `DMatrix` objects for optimized training.
- Best validation score achieved at **round 241**.
- **ROC-AUC (validation) ≈ 0.984**
- **ROC-AUC (test set) ≈ 0.974** ✅ best overall score

**XGBoost Feature Importance:** features **V17**, **V14**, **V12**, and **V10** dominate.

### 3.5 LightGBM

**Parameters (simple model):** `learning_rate=0.05`, `num_leaves=7`, `max_depth=4`, `scale_pos_weight=150` (class imbalance compensation), early stopping at 100 rounds.

#### Simple Training (train/valid split)

- Best round: **85** — **ROC-AUC (validation) ≈ 0.974**
- **ROC-AUC (test set) ≈ 0.946**

#### Cross-Validation (KFold, 5 folds)

The `LGBMClassifier` model is trained with cross-validation using finer hyperparameters (`n_estimators=2000`, `num_leaves=80`, L1/L2 regularization):

- Aggregated AUC on out-of-fold predictions: **~0.93**
- Test predictions are averaged across the 5 folds.

**LightGBM Feature Importance:** similar to XGBoost — **V14**, **V17**, **V12** are the most discriminative features.

---

## 4. Performance Summary

| Model | ROC-AUC | Evaluation Set |
|---|---|---|
| `RandomForestClassifier` | 0.85 | Validation |
| `AdaBoostClassifier` | 0.83 | Validation |
| `CatBoostClassifier` | 0.86 | Validation |
| `XGBoost` | **0.984** | Validation |
| `XGBoost` | **0.974** | **Test** ✅ |
| `LightGBM` (simple) | 0.974 | Validation |
| `LightGBM` (simple) | 0.946 | **Test** |
| `LightGBM` (cross-val) | 0.93 | Cross-validation |

---

## 5. Conclusions and Observations

### Best Performing Models

**XGBoost** and **LightGBM** significantly outperform the classical models (`RandomForest`, `AdaBoost`, `CatBoost`) on this fraud detection problem, achieving **ROC-AUC scores above 0.97** on the test set.

### Handling Class Imbalance

The `scale_pos_weight=150` parameter in LightGBM effectively compensates for the 99.83% / 0.17% class ratio in favor of legitimate transactions. This technique is preferable to simple resampling for boosting models.

### Feature Importance

PCA features **V17**, **V14**, **V12**, **V10**, and **V11** are consistently the most important across all boosting models. Features **V25**, **V26**, and **V28** have little predictive value.

### Early Stopping and Generalization

Using early stopping (50–100 rounds) is essential to prevent overfitting, especially on such imbalanced datasets. XGBoost stopped at round 241 (out of 1000 possible), indicating efficient learning without overfitting.

### Limitations and Future Work

- The dataset contains **no original features** (all anonymized by PCA), which limits business interpretability.
- Resampling techniques (SMOTE, undersampling) could be explored.
- Bayesian hyperparameter optimization (Optuna, Hyperopt) could further improve scores.

---

## 6. References

1. [Credit Card Fraud Detection — Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. [PCA — Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)
4. [LightGBM Paper (Microsoft Research)](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf)
5. [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
6. [CatBoost Documentation](https://catboost.ai/docs/)