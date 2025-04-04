import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Load dataset
def load_data():
    df = pd.read_csv("synthetic_maintenance_data_updated.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    return df

# Train ARIMA model
def train_arima(series):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    return model_fit

# Train XGBoost Classifier
def train_xgb(X_train, y_train):
    model = XGBClassifier(objective='binary:logistic', n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))  # Convert probabilities to 0/1
    return mae, rmse, accuracy

# Main Streamlit app
def main():
    st.title("Predictive Maintenance System")
    df = load_data()
    
    target_column = "Failure"
    features = ["Temperature", "Vibration", "Pressure", "Cycles"]
    X = df[features]
    y = df[target_column]

    # Split data first to prevent data leakage before applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Handling class imbalance using SMOTE only on training data
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # ARIMA Model
    st.write("## ARIMA Model")
    arima_model = train_arima(y_train)
    arima_pred = arima_model.forecast(steps=len(y_test))
    
    # Ensure predictions are aligned with y_test
    arima_pred = pd.Series(arima_pred, index=y_test.index)

    arima_mae, arima_rmse, _ = evaluate_model(y_test, arima_pred)
    st.write(f"ARIMA MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}")

    # XGBoost Model
    st.write("## XGBoost Model")
    xgb_model = train_xgb(X_train_resampled, y_train_resampled)
    xgb_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # Get probability of failure

    # Ensure predictions are aligned with y_test
    xgb_pred_prob = pd.Series(xgb_pred_prob, index=y_test.index)

    xgb_mae, xgb_rmse, xgb_acc = evaluate_model(y_test, xgb_pred_prob)
    st.write(f"XGBoost MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, Accuracy: {xgb_acc:.4f}")

    # Predict failure probabilities on full dataset
    df["Failure_Probability"] = xgb_model.predict_proba(X)[:, 1]

    st.write("## Predicted Failure Probabilities")
    st.write(df[["Timestamp", "Failure_Probability"]].tail(20))

    # Plot Results
    st.write("## Model Predictions vs Actual Failures")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual", color='black')
    ax.plot(arima_pred, label="ARIMA", linestyle='dashed', color='blue')
    ax.plot(xgb_pred_prob, label="XGBoost", linestyle='dashdot', color='orange')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
