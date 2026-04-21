# Predicting-hospital-readmissions-a-Machine-Learning-approach
This project applies classical machine learning techniques to predict whether a patient will be readmitted to hospital within 30 days of discharge, using a dataset of 5,000 patient encounters inspired by the work of Strack et al. The pipeline is split into two scripts: data preprocessing and ML modelling.

---

## Repository Structure

```
├── data_cleaning.py       # Data preprocessing and feature selection
├── ml_analysis.py         # Model training, evaluation, and plotting
└── README.md
```

---

## Dataset

The dataset contains 5,000 medical encounters with 37 features, including patient demographics, admission details, diagnoses (ICD-9 coded), medications, and the target variable `readmitted`. It is a subset of the [Diabetes 130-US Hospitals dataset](https://doi.org/10.24432/C5230J) from the UCI Machine Learning Repository.

---

## `data_cleaning.py` — Preprocessing & Feature Selection

The script prepares the raw dataset for modelling through the following steps:

**Decoding & grouping** — Columns encoded as integers (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) are decoded into meaningful categories and merged where groups are small (< 5% of data) to reduce dimensionality.

**NaN handling** — Missing values are replaced with random samples drawn from the same column's empirical distribution, preserving the original probability distribution rather than simply dropping rows.

**Feature removal** — Highly imbalanced features (> 90% identical value) and one uninformative column (`Unnamed: 0`) are dropped. The drug `tolbutamide` is also removed as 80% of its values were NaN.

**Re-encoding** — The `age` feature is converted from nominal ranges to ordinal numerical values. All remaining categorical features are then one-hot encoded, resulting in 52 features before selection.

**Feature selection** — A Random Forest classifier is used to rank features by two criteria: Mean Decrease in Impurity (MDI) and permutation-based accuracy decrease. The scores from both methods are combined into a single ranking. Model accuracy is then evaluated as a function of the number of features retained, revealing a plateau at around 20 features. The **top 20 features** are kept for modelling.

---

## `ml_analysis.py` — Modelling & Evaluation

Seven classifiers are trained and evaluated using **10-fold cross-validation**:

| Model | Optimal Parameter |
|---|---|
| K-Nearest Neighbors | k = 8 |
| Decision Tree | max depth = 4 |
| Random Forest | 31 trees, max depth = 4 |
| Naive Bayes | — |
| SVM | kernel = poly |
| Logistic Regression | C = 7.879 |
| XGBoost | 21 trees, max depth = 4 |

Each model is evaluated on three criteria: **accuracy**, **AUC**, and **total cost** using the asymmetric cost matrix below, which penalises false negatives (missed readmissions) much more heavily than false positives, reflecting the clinical stakes:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 0 | 50 |
| **Actual Negative** | 10 | 0 |

Computation time is also recorded for each model and parameter setting (averaged over 100 runs of 10-fold CV) to assess efficiency.

---

## Results

**XGBoost** and **Random Forest** achieve the best accuracy (~0.64) and AUC (~0.73 and ~0.69 respectively). From a cost perspective, XGBoost, Decision Tree, and Random Forest produce the lowest total misclassification cost. KNN performs worst across all metrics.

Overall, **XGBoost is the recommended model**, offering the best balance of accuracy, AUC, cost, and computational speed.

---

## Dependencies

```
pandas, numpy, scikit-learn, xgboost, matplotlib
```

---

## References

- Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Research International (2014)
- Clore et al., [Diabetes 130-US Hospitals Dataset](https://doi.org/10.24432/C5230J), UCI ML Repository (2014)

