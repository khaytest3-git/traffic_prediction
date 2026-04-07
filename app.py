from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "traffic_sample.csv"
LSTM_MODEL_CANDIDATES = [
    BASE_DIR / "lstm_model.h5",
    BASE_DIR / "lstm" / "lstm_model.h5",
]


def load_lstm_model():
    for model_path in LSTM_MODEL_CANDIDATES:
        if model_path.exists():
            try:
                from tensorflow.keras.models import load_model
            except ModuleNotFoundError:
                return None, None
            return load_model(model_path), model_path
    return None, None


# Load dataset
df = pd.read_csv(DATASET_PATH)

df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]
df = df.dropna()
df = df[df['SPEED'] > 0]

# Create congestion label
df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)


# Train Logistic Regression model
X = df[['HOUR', 'DAY_OF_WEEK']]
y = df['CONGESTION']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)


# Load trained LSTM model if available
lstm_model, lstm_model_path = load_lstm_model()

SEQUENCE_LENGTH = 6


# App title
st.title("Traffic Congestion Prediction")

st.write("This application predicts traffic congestion using both Logistic Regression and LSTM models.")


# Logistic Regression section
st.subheader("Logistic Regression Prediction")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Select Hour (0-23)", 0, 23, 8)

    day = st.selectbox(
        "Select Day of Week",
        [1,2,3,4,5,6,7],
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1]
    )

    predict_btn = st.button("Predict")

with col2:
    if predict_btn:
        input_data = pd.DataFrame({
            'HOUR': [hour],
            'DAY_OF_WEEK': [day]
        })

        prediction = lr_model.predict(input_data)[0]
        probability = lr_model.predict_proba(input_data)[0][1]

        st.caption("Prediction powered by Logistic Regression")

        st.metric("Congestion Probability", f"{probability:.2f}")

        if prediction == 1:
            st.error("High congestion expected")
        else:
            st.success("Low congestion expected")


# LSTM prediction using recent sequence
st.markdown("---")
st.subheader("LSTM Prediction (Recent Traffic Data)")

df['SPEED_DELTA'] = df['SPEED'].diff().fillna(0)
df['SPEED_ROLLING_MEAN'] = df['SPEED'].rolling(window=SEQUENCE_LENGTH, min_periods=1).mean()
df['MONTH'] = 1

features = ['SPEED', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'SPEED_DELTA', 'SPEED_ROLLING_MEAN']

sequence_data = df[features].values
scaler = MinMaxScaler()
sequence_data = scaler.fit_transform(sequence_data)

if lstm_model is None:
    st.warning("No trained LSTM model file was found. Save a trained model as `lstm_model.h5` to enable this section.")
else:
    latest_sequence = sequence_data[-SEQUENCE_LENGTH:]
    latest_sequence = latest_sequence.reshape(1, SEQUENCE_LENGTH, len(features))

    lstm_prob = lstm_model.predict(latest_sequence, verbose=0)[0][0]
    lstm_pred = 1 if lstm_prob >= 0.5 else 0

    st.caption(f"Loaded LSTM model from {lstm_model_path.name}")
    st.metric("LSTM Congestion Probability", f"{lstm_prob:.2f}")

    if lstm_pred == 1:
        st.error("High congestion predicted (LSTM)")
    else:
        st.success("Low congestion predicted (LSTM)")

st.caption("LSTM uses recent historical data for prediction")


# Chart
st.markdown("---")
st.subheader("Congestion Pattern by Hour")

hourly_congestion = df.groupby('HOUR')['CONGESTION'].mean()
st.line_chart(hourly_congestion)


# About section
st.markdown("---")
st.subheader("About the Model")

st.write("""
This application compares Logistic Regression and LSTM models for traffic congestion prediction.

Logistic Regression uses temporal features for direct prediction, while LSTM uses sequential traffic data to capture time-based patterns.
""")


# Dataset info
st.markdown("---")
st.subheader("Dataset Information")

st.write("""
The model is trained using historical traffic data from Chicago.

A sampled dataset is used in this deployed version to ensure efficient performance.
""")

