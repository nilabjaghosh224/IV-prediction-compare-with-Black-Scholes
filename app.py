# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy.stats import norm
from scipy.optimize import brentq

# --------------------------
# Black-Scholes functions
# --------------------------
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(option_price, S, K, T, r, option_type="call"):
    try:
        f = lambda sigma: (bs_call_price(S, K, T, r, sigma) - option_price) if option_type=="call" \
                          else (bs_put_price(S, K, T, r, sigma) - option_price)
        return brentq(f, 1e-6, 5.0)
    except:
        return np.nan

# --------------------------
# Load trained ANN model
# --------------------------
# Assume we have trained model saved as "iv_model.h5"
model = load_model("iv_model.h5")
scaler = StandardScaler()
# (⚠️ You must save & load the fitted scaler used during training!)

# --------------------------
# Streamlit App
# --------------------------
st.title("📈 Options Implied Volatility Predictor")
st.write("Compare Implied Volatility from ANN model vs. Black–Scholes")

# --- User Inputs for Features ---
st.sidebar.header("Market Features Input")
strikes_spread = st.sidebar.number_input("Strikes Spread", value=50.0)
calls_traded = st.sidebar.number_input("Calls Contracts Traded", value=1000)
puts_traded = st.sidebar.number_input("Puts Contracts Traded", value=800)
calls_oi = st.sidebar.number_input("Calls Open Interest", value=5000)
puts_oi = st.sidebar.number_input("Puts Open Interest", value=4500)
expirations_num = st.sidebar.number_input("Number of Expirations", value=5)
contracts_num = st.sidebar.number_input("Number of Contracts", value=20000)
hv_20 = st.sidebar.number_input("HV (20 days)", value=0.2)
hv_60 = st.sidebar.number_input("HV (60 days)", value=0.25)
hv_120 = st.sidebar.number_input("HV (120 days)", value=0.3)
vix = st.sidebar.number_input("VIX Index", value=20.0)

# --- User Inputs for Black-Scholes ---
st.sidebar.header("Black-Scholes Inputs")
S = st.sidebar.number_input("Spot Price (S)", value=4000.0)
K = st.sidebar.number_input("Strike Price (K)", value=4100.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=0.5)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.02)
market_price = st.sidebar.number_input("Market Option Price", value=200.0)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# --- ANN Prediction ---
features = np.array([[strikes_spread, calls_traded, puts_traded,
                      calls_oi, puts_oi, expirations_num, contracts_num,
                      hv_20, hv_60, hv_120, vix]])

features_scaled = scaler.fit_transform(features)   # ⚠️ In real use, load fitted scaler!
pred_iv = model.predict(features_scaled)[0]

# --- Black-Scholes IV ---
iv_bs = implied_vol(market_price, S, K, T, r, option_type)

# --------------------------
# Display Results
# --------------------------
st.subheader("🔮 Model Prediction")
iv_labels = ["DITM_IV", "ITM_IV", "sITM_IV", "ATM_IV", "sOTM_IV", "OTM_IV", "DOTM_IV"]
results_df = pd.DataFrame([pred_iv], columns=iv_labels)
st.write("**ANN Predicted Implied Volatilities**")
st.dataframe(results_df)

st.write("**Black–Scholes Implied Volatility**")
st.metric(label="IV from Black–Scholes", value=f"{iv_bs:.4f}")

# Compare visually
st.bar_chart(results_df.T)