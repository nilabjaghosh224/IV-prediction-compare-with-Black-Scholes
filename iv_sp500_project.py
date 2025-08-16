import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datasets import load_dataset

# Targets: IV buckets
TARGETS = ["DITM_IV", "ITM_IV", "sITM_IV", "ATM_IV", "sOTM_IV", "OTM_IV", "DOTM_IV"]

def load_data():
    dataset = load_dataset("gauss314/options-IV-SP500")
    df = pd.DataFrame(dataset["train"][:])
    df = df.sort_values("date")
    return df

def preprocess(df):
    y = df[TARGETS].values
    X = df.drop(columns=TARGETS + ["date", "symbol"]).fillna(0).values
    return X, y

def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, scalerX, scalery, epochs=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(X_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(criterion(model(xb), yb).item())
            print(f"Epoch {epoch}, Val Loss: {np.mean(val_losses):.4f}")

    return model

def evaluate(y_true, y_pred):
    metrics = {}
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        metrics[t] = {"MAE": mae, "RMSE": rmse}
    metrics["avg_MAE"] = np.mean([m["MAE"] for m in metrics.values() if isinstance(m, dict)])
    metrics["avg_RMSE"] = np.mean([m["RMSE"] for m in metrics.values() if isinstance(m, dict)])
    return metrics

def plot_results(y_true, y_pred, out_path="atm_iv_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:, 3], label="True ATM_IV")
    plt.plot(y_pred[:, 3], label="Predicted ATM_IV")
    plt.legend()
    plt.title("ATM_IV Prediction vs True")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

def main(mode, epochs, input_csv=None, output_csv=None):
    df = load_data()
    X, y = preprocess(df)
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)

    scalerX, scalery = StandardScaler(), StandardScaler()
    X_train = scalerX.fit_transform(X_train)
    X_val = scalerX.transform(X_val)
    X_test = scalerX.transform(X_test)
    y_train = scalery.fit_transform(y_train)
    y_val = scalery.transform(y_val)
    y_test = scalery.transform(y_test)

    if mode == "train":
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        joblib.dump(ridge, "ridge_model.pkl")
        torch.save(train_mlp(X_train, y_train, X_val, y_val, scalerX, scalery, epochs).state_dict(), "mlp_model.pt")
        joblib.dump((scalerX, scalery), "scalers.pkl")
        print("Models trained and saved.")

    elif mode == "eval":
        ridge = joblib.load("ridge_model.pkl")
        scalers = joblib.load("scalers.pkl")
        scalerX, scalery = scalers

        y_pred_ridge = scalery.inverse_transform(ridge.predict(X_test))
        metrics_ridge = evaluate(y_test, y_pred_ridge)
        print("Ridge Metrics:", metrics_ridge)

        model = MLP(X_train.shape[1], y_train.shape[1])
        model.load_state_dict(torch.load("mlp_model.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            y_pred_mlp = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        y_pred_mlp = scalery.inverse_transform(y_pred_mlp)
        metrics_mlp = evaluate(y_test, y_pred_mlp)
        print("MLP Metrics:", metrics_mlp)

        plot_results(y_test, y_pred_mlp)

    elif mode == "predict" and input_csv:
        new_df = pd.read_csv(input_csv)
        scalers = joblib.load("scalers.pkl")
        scalerX, scalery = scalers
        X_new, _ = preprocess(new_df.assign(**{t: 0 for t in TARGETS}))
        X_new = scalerX.transform(X_new)

        model = MLP(X_train.shape[1], y_train.shape[1])
        model.load_state_dict(torch.load("mlp_model.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_new, dtype=torch.float32)).numpy()
        y_pred = scalery.inverse_transform(y_pred)
        preds = pd.DataFrame(y_pred, columns=TARGETS)
        if output_csv:
            preds.to_csv(output_csv, index=False)
            print(f"Predictions saved to {output_csv}")
        else:
            print(preds.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval", "predict"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()
    main(args.mode, args.epochs, args.input_csv, args.output_csv)
