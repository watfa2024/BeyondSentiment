# Beyond Sentiment: Communication Risk Dynamics and Software Project Health

Advanced reproducibility package for the manuscript **Beyond Sentiment: Explaining Software Project Health through Communication Risk Dynamics**.

## Data

Place DATA in `data/`:

## 

## Install

```bash
pip install -r requirements.txt
```

Optional for extended analyses:

```bash
pip install xgboost shap
```

## Run

```bash
python run\_all.py
```

The pipeline validates data, builds CRI/SVI/CES/PHI constructs, runs descriptive statistics, construct validation, OLS theorem/proposition tests, ML algorithms, cross-validation, feature importance, optional SHAP, and exports figures/tables.

