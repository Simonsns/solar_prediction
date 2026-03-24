# Solar Power Forecasting — Occitanie (J+1)

![Python](https://img.shields.io/badge/Python-3.12-blue) ![LightGBM](https://img.shields.io/badge/LightGBM-multi--horizon-purple) ![PyTorch](https://img.shields.io/badge/PyTorch-seq2seq-orange) ![Status](https://img.shields.io/badge/status-in%20progress-yellow)

End-to-end MLOps pipeline for day-ahead solar power forecasting in the Occitanie region (France), using public RTE production data and Open-Meteo weather forecasts.

## Project Overview

This project forecasts regional solar power production (MW) at hourly resolution for horizons t+1 to t+24, leveraging open data sources and a multi-model architecture. The pipeline covers the full ML lifecycle: data ingestion, feature engineering, model training, and automated inference with drift-based retraining.

**Data sources**
- [RTE Open Data API](https://data.rte-france.com/) — regional solar production (hourly, 2023–2025)
- [Open-Meteo API](https://open-meteo.com/) — historical and forecast weather variables
- Installed capacity: RTE API (used to normalize the target variable)

**Spatial approach**: weather features are extracted at the barycentre of installed capacity per department, with spatial dispersion variables (min-max range, std) to capture regional heterogeneity without requiring a CNN.

## Repository Structure *(work in progress)*

```
solar_prediction/
│
├── data/
│   ├── raw/                            # Raw data
│   ├── external/                       # External data (from API)
│   └── processed/                      # Training datasets, feature lists
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA : STL/MSTL decomposition, ACF/PACF, CCF analysis
│   ├── 02_feature_engineering.ipynb    # Cyclic encoding, lags, rolling windows, outlier guards
│   └── 03_model_training.ipynb         # SARIMAX baseline, LightGBM multi-horizon, seq2seq LSTM
│
├── src/
│   ├── data_pipeline/
│   │   ├── data_collection/
│   │   │   ├── solar/fetching_data.py  # RTE API ingestion
│   │   │   └── weather/fetching_data.py # Open-Meteo historical + forecast
│   │   ├── data_processing/            # Preprocessing, feature engineering
│   │   └── run_etl.py                  # Orchestrated ETL pipeline
│   └── config.py
│
├── models/mlflow_artifacts             # Serialized model bundles 
├── logs/
└── tests/                              # (in progress)
```

## Methodology

### Target variable
Solar production normalized by installed capacity (`solar_mw / chronique_capacity`), which removes the long-term upward trend due to new installations — confirmed by MSTL decomposition.

### Feature engineering
- **Cyclic encoding**: hour and month (sin/cos) to capture solar cycle seasonality
- **Lagged features**: lags at t-24, t-36, t-48, t-60, t-72 (justified by CCF analysis)
- **Rolling statistics**: mean, std, ramp over 6, 12, 24, 48 periods
- **Spatial dispersion**: min-max range and std across departmental barycentres
- **Horizon-specific feature selection**: cumulative gain method (LightGBM embedding, 95% threshold) applied independently for short (t+1→t+3), mid (t+4→t+12), and long (t+13→t+24) horizons

### Models

| Model | Role | Horizon |
|---|---|---|
| SARIMAX(1,0,q)(0,1,Q)₂₄ | Statistical baseline | t+1 |
| LightGBM × 24 | Main production model | t+1 → t+24 |
| Seq2seq LSTM (Encoder-Decoder) | Deep learning benchmark | t+1 → t+24 |

**LightGBM strategy**: one model per horizon with independent Optuna HP optimisation (`TimeSeriesSplit`, 5 folds, 50 trials). All 24 models are serialized as a single bundle with the fitted scaler and PSI reference distribution.

**Seq2seq architecture**: LSTM Encoder processes the past sequence (X_past), Decoder takes future weather forecasts (X_future) as input — physically motivated since Open-Meteo provides J+1 forecasts at inference time. LayerNorm on hidden states for training stability.

### Validation
- Temporal train/test split (85/15), no shuffling
- Nested cross-validation for model selection (outer: 5 folds, inner: 3 folds)
- Backtest walk-forward on the last 90 days (in progress)
- Baseline comparison: naïve persistence (y_t = y_{t-24})

## MLOps Architecture

```
Open-Meteo / RTE APIs
        ↓
AWS Lambda (every 4h)          ← orchestration
        ↓
   PSI monitoring              ← drift detection
   ├── PSI > 0.25 → retrain → push bundle to R2
   └── PSI ≤ 0.25 → load existing bundle from R2
        ↓
   predict J+1 → store latest.json on R2
        ↓
MLflow (HF Spaces)             ← experiment tracking
GitHub Actions                 ← CI/CD (tests + deploy)
```

**Storage**: Currently on Supabase. In the future : Cloudflare R2 (S3-compatible, free tier) for model artefacts and predictions. Boto3 code is directly portable to AWS S3 by changing `endpoint_url`.

## Current Status

| Component | Status |
|---|---|
| ETL pipeline | Done |
| EDA | Done |
| Feature engineering | Done |
| LightGBM multi-horizon training | Done |
| Seq2seq LSTM | Done |
| SARIMAX baseline | In progress |
| Test set evaluation + backtest | In progress |
| Pydantic config (BaseSettings) | To do |
| Unit tests | To do |
| Lambda + R2 deployment | To do |
| MLflow tracking | To do |
| CI/CD GitHub Actions | To do |

## Key Technical Choices

**Why 24 separate LightGBM models?** Each horizon has a different feature importance structure — autoregressions dominate at t+1 while weather forecasts become critical at t+24. A single multi-output model would force the same HP across all horizons, losing this horizon-specific signal.

**Why normalize by installed capacity?** Solar production has a long-term upward trend driven by new installations (~+15% over 2023–2025 in Occitanie). Normalizing removes this non-stationarity without differencing, preserving the interpretability of the target.

**Why spatial dispersion features?** Cloud cover and DNI are highly heterogeneous across Occitanie's departments (altitude differences of 400m+, Mediterranean influence). Departmental min-max range and std capture this spatial variance without requiring a CNN.

## Requirements

```
python >= 3.11
lightgbm, torch, statsmodels, pmdarima
scikit-learn, optuna
pandas, numpy, geopandas
openmeteo-requests, requests-cache, retry-requests
boto3, pydantic-settings
mlflow
```

---

## Author

Simon — Data Scientist
[GitHub](https://github.com/Simonsns)
