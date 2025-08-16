# train_iv_model.py

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load Dataset
id = "gauss314/options-IV-SP500"
data_iv = load_dataset(id)
df = pd.DataFrame(data_iv['train'][:])

# 2. Features & Targets
target_cols = ["DITM_IV", "ITM_IV", "sITM_IV", "ATM_IV", "sOTM_IV", "OTM_IV", "DOTM_IV"]
feature_cols = [col for col in df.columns if col not in target_cols]

X = df[feature_cols].fillna(0).values
y = df[target_cols].fillna(0).values

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
import joblib
joblib.dump(scaler, "scaler.pkl")

# 5. Build ANN
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(target_cols), activation="linear")  # 7 outputs
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 6. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    verbose=1
)

# 7. Save Model
model.save("iv_model.h5")
print("✅ Model trained & saved as iv_model.h5")