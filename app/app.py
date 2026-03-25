from pathlib import Path
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed"
X_PATH = DATA_PATH / "X_train.csv"
Y_PATH = DATA_PATH / "y_train.csv"

# ----------------------------
# File existence checks
# ----------------------------
required_files = [MODEL_PATH, X_PATH, Y_PATH]
missing = [str(p) for p in required_files if not p.exists()]

if missing:
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join(missing)
        + "\n\nRun 04_modeling.ipynb first."
    )

# ----------------------------
# Load model and data
# ----------------------------
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(X_PATH)
y_train = pd.read_csv(Y_PATH).squeeze()

# Ensure y_train is a 1D Series
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Olist Return Probability Predictor", layout="centered")
st.title("Olist Return Probability Predictor")

st.markdown(
    """
    Enter order-related information to estimate the probability of return.
    """
)

# ----------------------------
# User inputs
# ----------------------------
st.header("User Inputs")

review_score = st.slider("Review Score", min_value=1, max_value=5, value=3, step=1)
delivery_late = st.selectbox("Was the delivery late?", [0, 1], index=0)
total_price = st.number_input(
    "Total Price", min_value=0.0, max_value=100000.0, value=100.0, step=1.0
)

# Match model feature order exactly
expected_features = list(X_train.columns)

input_df = pd.DataFrame(
    [[review_score, delivery_late, total_price]],
    columns=["review_score", "delivery_late", "total_price"],
)

# Reorder to match training feature order
input_df = input_df.reindex(columns=expected_features)

# ----------------------------
# Prediction
# ----------------------------
st.header("Prediction Result")

prediction_proba = None
prediction_class = None

if st.button("Predict Return Probability"):
    if hasattr(model, "predict_proba"):
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction_class = model.predict(input_df)[0]
        st.success(f"Predicted return probability: **{prediction_proba * 100:.2f}%**")
        st.write(f"Predicted class: **{prediction_class}**")
    else:
        prediction_class = model.predict(input_df)[0]
        st.success(f"Predicted return class: **{prediction_class}**")

# ----------------------------
# Model performance
# ----------------------------
st.header("Model Accuracy on Training Set")
train_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, train_pred)
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

# ----------------------------
# SHAP explanation
# ----------------------------
st.header("Feature Importance (SHAP Explanation)")

try:
    # Use unified SHAP explainer
    explainer = shap.Explainer(model, X_train)

    # -------- Global explanation --------
    st.subheader("Global Feature Importance")

    X_sample = X_train.sample(n=min(300, len(X_train)), random_state=42)
    shap_values_global = explainer(X_sample)

    # If explanation has class dimension, select the positive class (class 1)
    if len(shap_values_global.shape) == 3:
        shap_values_global = shap_values_global[:, :, 1]

    plt.figure()
    shap.plots.beeswarm(shap_values_global, show=False)
    fig_global = plt.gcf()
    st.pyplot(fig_global, clear_figure=True)

    st.caption(
        "This global SHAP plot shows how important each feature is across many training examples. "
        "Features with larger SHAP value spread have stronger overall influence on the model."
    )

    # -------- Local explanation --------
    st.subheader("Local Explanation for Current Input")

    st.write("Current input used for prediction:")
    st.dataframe(input_df, use_container_width=True)

    shap_values_local = explainer(input_df)

    # If explanation has class dimension, select class 1
    if len(shap_values_local.shape) == 3:
        shap_values_local = shap_values_local[:, :, 1]

    plt.figure()
    shap.plots.waterfall(shap_values_local[0], show=False)
    fig_local = plt.gcf()
    st.pyplot(fig_local, clear_figure=True)

    st.caption(
        "This local SHAP waterfall plot explains how each feature pushed the prediction higher or lower "
        "for the current input."
    )

except Exception as e:
    st.warning(f"SHAP explanation could not be generated: {e}")