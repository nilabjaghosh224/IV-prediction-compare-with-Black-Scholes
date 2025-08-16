# Options IV S&P500 Project

This project predicts **implied volatilities (IVs)** for S&P500 options (DITM, ITM, sITM, ATM, sOTM, OTM, DOTM) using the Hugging Face dataset `gauss314/options-IV-SP500`.

## Features
- Ridge Regression baseline
- PyTorch MLP (multi-output regression)
- Time-based train/val/test split
- Evaluation: MAE & RMSE per IV bucket
- ATM_IV prediction plot
- Batch prediction on new CSV

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install datasets pandas numpy scikit-learn torch matplotlib joblib
```

## Usage
Train models:
```bash
python iv_sp500_project.py --mode train --epochs 50
```

Evaluate models:
```bash
python iv_sp500_project.py --mode eval
```

Predict on new data:
```bash
python iv_sp500_project.py --mode predict --input_csv your.csv --output_csv preds.csv
```

## Output
- ridge_model.pkl, mlp_model.pt → trained models
- scalers.pkl → data scalers
- atm_iv_plot.png → prediction chart
- preds.csv → predictions (when using --mode predict)
