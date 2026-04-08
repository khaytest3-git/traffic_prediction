from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "traffic_sample.csv"
LSTM_MODEL_PATH = BASE_DIR / "lstm_model.h5"
SEQUENCE_LENGTH = 6


def load_lstm_model():
    try:
        from tensorflow.keras.models import load_model
        if not LSTM_MODEL_PATH.exists():
            return None
        return load_model(LSTM_MODEL_PATH)
    except Exception:
        return None


df = pd.read_csv(DATASET_PATH)

df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]
df = df.dropna()
df = df[df['SPEED'] > 0]

df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)


X = df[['HOUR', 'DAY_OF_WEEK']]
y = df['CONGESTION']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)


st.title("Traffic Congestion Prediction")

st.write("This application predicts traffic congestion based on time patterns using Logistic Regression, Gradient Boosting, and LSTM models.")


st.subheader("Prediction")

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

        st.caption("Logistic Regression")

        st.metric("Congestion Probability (LR)", f"{probability:.2f}")

        if prediction == 1:
            st.error("High congestion expected")
        else:
            st.success("Low congestion expected")


st.markdown("---")
st.subheader("Gradient Boosting Prediction")

if predict_btn:
    input_data = pd.DataFrame({
        'HOUR': [hour],
        'DAY_OF_WEEK': [day]
    })

    gb_prediction = gb_model.predict(input_data)[0]
    gb_probability = gb_model.predict_proba(input_data)[0][1]

    st.caption("Gradient Boosting")

    st.metric("Congestion Probability (GB)", f"{gb_probability:.2f}")

    if gb_prediction == 1:
        st.error("High congestion expected (GB)")
    else:
        st.success("Low congestion expected (GB)")


lstm_model = load_lstm_model()

st.markdown("---")
st.subheader("LSTM Prediction (Recent Data Sequence)")

if lstm_model is not None:
    df['SPEED_DELTA'] = df['SPEED'].diff().fillna(0)
    df['SPEED_ROLLING_MEAN'] = df['SPEED'].rolling(window=3).mean().bfill()

    features = ['SPEED', 'HOUR', 'DAY_OF_WEEK', 'SPEED_DELTA', 'SPEED_ROLLING_MEAN']

    sequence_data = df[features].values
    latest_sequence = sequence_data[-SEQUENCE_LENGTH:]
    latest_sequence = latest_sequence.reshape(1, SEQUENCE_LENGTH, len(features))

    lstm_prob = lstm_model.predict(latest_sequence)[0][0]
    lstm_pred = 1 if lstm_prob >= 0.5 else 0

    st.metric("Congestion Probability (LSTM)", f"{lstm_prob:.2f}")

    if lstm_pred == 1:
        st.error("High congestion predicted (LSTM)")
    else:
        st.success("Low congestion predicted (LSTM)")

    st.caption("LSTM uses historical sequence data")

else:
    st.warning("LSTM model unavailable in this deployment environment.")


st.markdown("---")
st.subheader("Congestion Pattern by Hour")

hourly_congestion = df.groupby('HOUR')['CONGESTION'].mean()
st.line_chart(hourly_congestion)


st.markdown("---")
st.subheader("About the Model")

st.write("""
This dashboard predicts traffic congestion based on temporal patterns.

Logistic Regression and Gradient Boosting use time-based features such as hour and day of week.

LSTM is used to capture sequential patterns from historical traffic data.
""")


st.markdown("---")
st.subheader("Dataset Information")

st.write("""
The model is trained using historical traffic data from Chicago.

Congestion is defined based on speed thresholds, but speed is not used as an input feature in the prediction models to ensure meaningful pattern learning.
""")

