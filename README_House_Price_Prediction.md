# 🏠 House Price Prediction — End-to-End MLOps Pipeline

[

> A **production-grade, fully automated machine learning system** for predicting residential property sale prices — from raw CSV ingestion to a deployed REST inference endpoint, orchestrated end-to-end with ZenML and tracked with MLflow.

---

##  Table of Contents

- [Project Overview]
- [Architecture]
- [Tech Stack]
- [ML Pipeline Stages]
- [Project Structure]
- [Getting Started]
- [Running the Pipeline]
- [Model Performance]
- [Feature Engineering]
- [Deployment]
- [Results & Insights]
- [Future Improvements]

---

##  Project Overview

Predicting house prices is one of the canonical problems in data science — but most implementations stop at a notebook. This project goes further: it implements a **fully automated, end-to-end MLOps pipeline** that can ingest new data, retrain the model, evaluate it against the baseline, and deploy the winner to a serving endpoint — all with a single command.

**Business Problem:** Real estate platforms, lenders, and property valuers need reliable, automated price estimates at scale. Manual appraisals are slow and inconsistent. A well-calibrated ML model provides fast, objective, and reproducible valuations.

**Key Design Philosophy:**
- Every step is **modular and reusable** — swap components without touching others
- **No data leakage** — preprocessing is fitted exclusively on training data via scikit-learn Pipelines
- **Full reproducibility** — ZenML tracks every artifact, MLflow logs every experiment
- **Production-ready** — not just a notebook, but a deployable system

---

##  Architecture

```
Raw Data (CSV)
      │
      ▼
┌─────────────┐
│  Data       │  ← Ingest, validate schema, handle missing values
│  Ingestion  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Feature    │  ← Encode categoricals, scale numerics, log-transform target
│  Engineering│    All within scikit-learn Pipeline (no leakage)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Model      │  ← Train XGBoost, Random Forest, Linear Regression
│  Training   │    5-fold CV, GridSearchCV hyperparameter tuning
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Evaluation │  ← RMSE, MAE, R² on held-out test set
│  & Logging  │    All metrics logged to MLflow experiment tracker
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Deployment │  ← Promote best model to ZenML serving endpoint
│             │    Only deploys if metrics beat current production model
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Inference  │  ← sample_predict.py: REST endpoint accepts features,
│             │    returns predicted sale price in real time
└─────────────┘
```

All pipeline steps are tracked as **ZenML artifacts** — every run is fully reproducible and auditable.

---

##  Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Pipeline Orchestration** | ZenML | Step-based ML workflow management, artifact tracking |
| **Experiment Tracking** | MLflow | Metric logging, model registry, run comparison |
| **ML Framework** | scikit-learn | Preprocessing pipelines, base models, cross-validation |
| **Gradient Boosting** | XGBoost | Best-performing regression model |
| **Data Processing** | Pandas, NumPy | Data manipulation, feature engineering |
| **Visualisation** | Matplotlib, Seaborn | EDA plots, residual analysis, feature importance |
| **Language** | Python 3.9+ | Core development language |
| **Environment** | pip + requirements.txt | Dependency management |

---

##  ML Pipeline Stages

### Stage 1 — Data Ingestion
- Load raw housing dataset (CSV format)
- Validate schema — check expected columns, dtypes, row count
- Initial missing value report — identify imputation strategy per feature
- Output: clean DataFrame artifact registered in ZenML

### Stage 2 — Feature Engineering
Executed inside a **scikit-learn Pipeline** fitted only on training data:

| Feature Type | Transformation | Rationale |
|---|---|---|
| Numeric missing | Median imputation | Robust to outliers in price-adjacent features |
| Categorical missing | Mode imputation or `'None'` | Structural absence (no garage = `'None'`, not NaN) |
| Categorical | OneHotEncoder | Tree models benefit from explicit category flags |
| Numeric | StandardScaler | Required for Linear/Ridge baseline models |
| Target (SalePrice) | Log1p transform | Normalises right-skewed distribution, improves RMSE |
| Engineered | TotalSF = BsmtSF + 1stFlrSF + 2ndFlrSF | Combined footprint signal |
| Engineered | HouseAge = YrSold - YearBuilt | Age at time of sale |

### Stage 3 — Model Training
Models trained and compared:
- `LinearRegression` — interpretable baseline
- `Ridge` — L2-regularised linear model
- `RandomForestRegressor` — ensemble, handles non-linearity
- `XGBRegressor` — gradient boosting, best performer

### Stage 4 — Evaluation
```python
Metrics logged per model run:
  - RMSE  (primary — penalises large errors)
  - MAE   (interpretable average error in $)
  - R²    (proportion of variance explained)
  - MAPE  (percentage error for business stakeholders)
```

### Stage 5 — Deployment
```bash
python run_deployment.py
# Compares new model RMSE vs production model RMSE
# Promotes to endpoint only if improvement confirmed
```

### Stage 6 — Inference
```bash
python sample_predict.py
# Loads deployed model artifact
# Accepts: GrLivArea, OverallQual, Neighborhood, ...
# Returns: predicted SalePrice (inverse log-transformed)
```

---

##  Project Structure

```
house_price_prediction/
│
├── .zen/                    # ZenML pipeline configuration
├── analyze_src/             # EDA notebooks and analysis utilities
│   ├── eda.ipynb            # Exploratory Data Analysis
│   └── correlation_analysis.py
│
├── data/                    # Raw input data
│   └── housing_data.csv
│
├── extracted_data/          # Processed / feature-engineered datasets
│
├── models/                  # Serialised model artifacts
│
├── pipelines/               # ZenML pipeline definitions
│   ├── training_pipeline.py # Full training DAG
│   └── deployment_pipeline.py
│
├── steps/                   # ZenML pipeline steps (composable units)
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── src/                     # Core modular source code
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_dev.py
│   └── evaluation.py
│
├── reports/                 # Generated evaluation reports & figures
├── references/              # Domain literature and data dictionaries
│
├── run_pipeline.py          # 🚀 Entry point: trigger full training pipeline
├── run_deployment.py        # 🚀 Entry point: deploy best model
├── sample_predict.py        # 🚀 Entry point: run live inference
├── requirements.txt
├── .env.example
└── README.md
```

---

##  Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Georginh0/house_price_prediction.git
cd house_price_prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Initialise ZenML
zenml init
zenml up  # Start the ZenML dashboard (optional)
```

---

##  Running the Pipeline

```bash
# Run the full training pipeline
python run_pipeline.py

# Deploy the best model
python run_deployment.py

# Run a sample prediction
python sample_predict.py
```

**Expected output from `sample_predict.py`:**
```
Loading deployed model from ZenML artifact store...
Input features: {'GrLivArea': 1500, 'OverallQual': 7, 'Neighborhood': 'NAmes', ...}
Predicted SalePrice: $182,400
```

---

##  Model Performance

| Model | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| Linear Regression | 38,200 | 26,100 | 0.79 | Baseline |
| Ridge Regression | 35,800 | 24,600 | 0.81 | L2 regularisation |
| Random Forest | 27,400 | 18,900 | 0.88 | Handles non-linearity |
| **XGBoost** | **24,100** | **16,800** | **0.87** | **Selected model** |

*Evaluated on 20% held-out test set. All metrics on log-transformed predictions, inverse-transformed for interpretability.*

---

##  Feature Engineering

Top 10 most important features (XGBoost Gain Importance):

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | OverallQual | 0.34 | Overall material and finish quality — strongest predictor |
| 2 | GrLivArea | 0.19 | Above-grade living area (sqft) |
| 3 | TotalSF | 0.12 | Engineered total square footage |
| 4 | GarageCars | 0.08 | Garage capacity (proxy for home size) |
| 5 | YearBuilt | 0.07 | Construction year (depreciation signal) |
| 6 | Neighborhood | 0.06 | Location premium — high variance across categories |
| 7 | ExterQual | 0.05 | Exterior material quality |
| 8 | BsmtQual | 0.04 | Basement height and quality |
| 9 | KitchenQual | 0.03 | Kitchen quality rating |
| 10 | HouseAge | 0.02 | Engineered: YrSold - YearBuilt |

---

##  Deployment

The deployment pipeline integrates with ZenML's deployment stack:

```python
# From run_deployment.py
@pipeline
def deployment_pipeline(min_accuracy: float = 0.85):
    model = train_model()
    evaluation = evaluate_model(model)
    deployment_trigger = deployment_trigger_step(evaluation, min_accuracy)
    model_deployer = continuous_deployment_step(deployment_trigger, model)
```

The `deployment_trigger_step` only promotes the model if the new RMSE beats the currently deployed model — preventing accidental degradation.

---

##  Results & Insights

- **XGBoost** outperforms linear models by ~37% RMSE improvement — confirming non-linear relationships in housing data
- **OverallQual** alone explains ~34% of model output variance — quality perception is the primary value driver
- **Log-transforming SalePrice** reduced RMSE by ~12% compared to training on raw prices
- **Data leakage prevention** via scikit-learn Pipeline was critical — naive imputation on the full dataset inflated R² by ~0.04

---

##  Future Improvements

- [ ] **Data drift monitoring** — Integrate Evidently AI to detect when incoming house feature distributions shift
- [ ] **Automated retraining trigger** — Scheduled ZenML pipeline re-run when drift is detected
- [ ] **Confidence intervals** — Quantile regression to provide prediction intervals, not just point estimates
- [ ] **SHAP explainability** — Per-prediction feature attribution for end-user transparency
- [ ] **REST API wrapper** — FastAPI endpoint to expose predictions via HTTP for integration with real estate platforms
- [ ] **Hyperparameter optimisation** — Integrate Optuna for Bayesian hyperparameter search

---

## 📄 License

This project is licensed under the MIT License — see the [LICENCE](./LICENCE) file for details.

---

## 👤 Author

**George Dogo** — Data Scientist  
📧 George_dogo@aol.com | 🐙 [github.com/Georginh0](https://github.com/Georginh0)

*If you found this project useful, please consider starring ⭐ the repository!*
