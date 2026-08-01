# Natural Disaster Severity Classification

A multi-method machine learning project classifying the severity of natural disaster events
using NOAA Storm Events data (2000–2025) across five model families and 13 experiments in Python.

## Project Overview

This project was completed as part of STA6704 Data Mining Methodology II at the University of
Central Florida. The goal was to build and compare machine learning models capable of classifying
natural disaster severity into five ordinal classes — None, Low, Medium, High, and Catastrophic —
to support proactive emergency resource allocation. Predicting which events are likely to reach
catastrophic impact before they occur could enable emergency managers to pre-position resources
rather than deploying them reactively.

The dataset presented a structural challenge common to real-world disaster records: severe events
are extraordinarily rare. High and Catastrophic severity events together represent fewer than
0.05% of all records, making standard classification approaches prone to collapsing predictions
onto the dominant None and Low classes. The project systematically evaluated five model families
across 13 experiments, varying feature sets, imbalance handling strategies, and hyperparameter
configurations, to assess both predictive ceiling and the sources of classification failure.

## Dataset

- **Source:** NOAA Storm Events Database, 2000–2025 (publicly available at
  [ncdc.noaa.gov](https://www.ncdc.noaa.gov/stormevents/))
- **Raw:** 3,777,888 rows across 43 event types
- **After preprocessing:** 1,546,474 unique events, 31 columns
- **Response variable:** `SEVERITY_CLASS` (None / Low / Medium / High / Catastrophic)
- **Class imbalance:** 75.3% None, 23.3% Low, 1.4% Medium, 0.03% High, 0.01% Catastrophic
- **Split strategy:** 70% training / 15% validation / 15% test (stratified, random state = 6704)

## Preprocessing

- Removed duplicate rows via exact deduplication (3.77M → 1.55M rows)
- Dropped FEMA join records after type-validation revealed only 7% valid event-type matches
- Dropped latitude/longitude (41% structurally missing — forecast-zone records never have coordinates)
- Constructed `DAMAGE_TOTAL_USD` by combining property and crop damage; NaN retained where
  both sources were missing (genuinely unknown, not zero)
- Labeled offshore/marine zone records as `STATE_ABBREV = MARINE`
- Engineered `MAGNITUDE_APPLICABLE` flag to distinguish structural from informative missingness
  in the MAGNITUDE column
- Constructed target variable `SEVERITY_CLASS` from a composite score aggregating deaths,
  injuries, and property damage with ordinal thresholds grounded in the empirical training
  distribution
- Removed all death, injury, and damage columns post-selection as post-event leakage

## Feature Selection

A three-method consensus pipeline was applied to identify a parsimonious, leakage-free feature set:

| Method | Approach | Features Selected |
|---|---|---|
| Filter | VarianceThreshold (threshold = 0.01) | Removed 50 near-zero-variance features |
| Wrapper | Bidirectional stepwise OLS (p_in = 0.01, p_out = 0.05) | 99 features (~198 min runtime) |
| Embedded | LassoCV (α = 0.0084) | 58 features retained |
| Embedded | ElasticNetCV (α = 0.0169, l1_ratio = 0.50) | 58 features retained |
| **Consensus** | **≥ 2 of 3 methods → leakage removal** | **51 final features** |

Top predictors by Lasso coefficient magnitude: `DAMAGE_TOTAL_USD_REPORTED` (0.284),
`MAGNITUDE_TYPE_EG` (0.120), `EVENT_TYPE_Tornado` (0.099).

## Methods

### Model Families and Experiments

| # | Model | Experiment | Val Macro F1 | Test Macro F1 |
|---|---|---|---|---|
| 1 | Random Forest | Baseline (51 features, balanced) | 0.3496 | 0.3488 |
| 2 | Random Forest | Remove YEAR (temporal artifact) | 0.2981 | — |
| 3 | Random Forest | SMOTE oversampling | 0.3636 | — |
| 4 | **Random Forest** | **Full pre-selection (109 features)** | **0.3988** | **0.3966** |
| 5 | GLM | Baseline (multinomial, no penalty) | 0.3121 | — |
| 6 | GLM | Elastic Net (l1_ratio=0.5, C=1.0) | 0.3042 | 0.3040 |
| 7 | XGBoost | Baseline (depth=6, lr=0.1) | 0.3268 | — |
| 8 | XGBoost | Tuned (depth=10, lr=0.05, n_est=73) | 0.3355 | 0.3345 |
| 9 | SVM | LinearSVC (full training set) | 0.3018 | 0.3021 |
| 10 | SVM | LinearSVC (None undersampled →100K) | 0.2796 | 0.2804 |
| 11 | SVM | RBF SVC (100K stratified sample) | 0.3124 | 0.3166 |
| 12 | Neural Network | sklearn MLP (250K sample, best) | 0.3380 | 0.3375 |
| 13 | Neural Network | PyTorch MLP (256/128, Tanh, LBFGS) | 0.3306 | 0.3298 |

All experiments used macro-averaged F1 as the primary metric to weight all five severity
classes equally. YEAR was identified as a reporting coverage artifact (Medium severity declined
steadily 2000–2025 due to NOAA database expansion, not real trends). SMOTE was rejected after
inspection showed gains driven entirely by the None class with High and Catastrophic collapsing
to near-zero recall.

## Results

| Model Family | Best Experiment | Test Macro F1 |
|---|---|---|
| **Random Forest** | Full 109-feature set | **0.3966** |
| Neural Network | sklearn MLP 250K | 0.3375 |
| XGBoost | Tuned config 4 | 0.3345 |
| SVM | RBF SVC 100K | 0.3166 |
| GLM | Elastic Net | 0.3040 |

Across all model families, None and Low severity events were reliably classified (Low F1
consistently exceeding 0.58), while High and Catastrophic events remained near-zero F1
regardless of model family, imbalance handling strategy, or feature set. XGBoost achieved
the highest ROC AUC (0.9113) despite modest macro F1, indicating strong ranking ability
even where hard classification failed.

This consistent ceiling reflects a fundamental feature informativeness limitation: the NOAA
Storm Events database records what happened and where, but does not capture population density,
infrastructure quality, or pre-event preparedness capacity — the variables that plausibly
discriminate catastrophic from non-catastrophic outcomes.

## Project Structure
NaturalDisasterSeverityClassification/
    - README.md
    - Natural_Disaster_Impact_Modelling_Final_Code_Final_Draft.ipynb # Full pipeline notebook
    - Final_Report_IEEE.docx # IEEE Transactions paper
    - data/
        - README.md # Data download instructions

## Setup

1. Clone this repo
2. Download the NOAA Storm Events database (2000–2025) from
   [ncdc.noaa.gov/stormevents](https://www.ncdc.noaa.gov/stormevents/choosedates.jsp)
   or access the merged dataset from the shared Google Drive folder linked in the notebook
3. Install required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost torch joblib statsmodels
```

4. Open `Natural_Disaster_Impact_Modelling_Final_Code_Final_Draft.ipynb` and run cells
   sequentially from top to bottom
5. Note: the bidirectional stepwise selection cell (~198 min) and RBF SVC full training cell
   (several hours) are pre-loaded with results and commented out by default — do not uncomment
   unless you intend to re-run them

## Team

| Member | Contribution |
|---|---|
| Alejandro Arroyo | Project architecture; data pipeline and cleaning; feature selection pipeline (stepwise, Lasso, ElasticNet); Random Forest (4 experiments); SVM (3 experiments); notebook consolidation; IEEE paper; master results comparison |
| Ruby Penaranda Espinoza | Exploratory data analysis (8 visualizations) |
| Sanjana Vijayabhaskar | XGBoost (3 experiments, hyperparameter search, 5-fold CV) |
| Akila Kumar | GLM / Multinomial Logistic Regression (2 experiments, coefficient analysis) |
| Harris Durani | SVM pipeline; Neural Network (sklearn MLP scaling study + PyTorch GPU implementation) |

## Course

STA6704 Data Mining Methodology II. University of Central Florida, Summer 2026.
