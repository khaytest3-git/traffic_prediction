from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent
LSTM_MODEL_PATH = BASE_DIR / "lstm_model.h5"
LSTM_SCALER_PATH = BASE_DIR / "lstm_scaler.joblib"
LSTM_FEATURES_PATH = BASE_DIR / "lstm_feature_columns.joblib"
LR_MODEL_PATH = BASE_DIR / "lr_model.joblib"
GB_MODEL_PATH = BASE_DIR / "gb_model.joblib"
SEQUENCE_LENGTH = 6

CANDIDATE_DATASETS = [
    BASE_DIR / "Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv",
    BASE_DIR / "traffic_sample.csv",
]


@st.cache_resource
def load_lstm_model():
    try:
        from tensorflow.keras.models import load_model
        if not LSTM_MODEL_PATH.exists():
            return None
        return load_model(LSTM_MODEL_PATH)
    except Exception:
        return None


@st.cache_resource
def load_lstm_assets():
    if not LSTM_SCALER_PATH.exists() or not LSTM_FEATURES_PATH.exists():
        return None, None
    scaler = joblib.load(LSTM_SCALER_PATH)
    feature_columns = joblib.load(LSTM_FEATURES_PATH)
    return scaler, feature_columns


def build_lstm_sequence(df, hour, day, scaler, feature_columns):
    speed_lookup = df.groupby(['HOUR', 'DAY_OF_WEEK'])['SPEED'].mean()
    global_mean_speed = df['SPEED'].mean()

    rows = []
    for i in range(SEQUENCE_LENGTH):
        h = (hour - SEQUENCE_LENGTH + 1 + i) % 24
        speed = speed_lookup.get((h, day), global_mean_speed)
        row = {'SPEED': speed, 'HOUR': h, 'DAY_OF_WEEK': day}
        if 'MONTH' in feature_columns:
            row['MONTH'] = pd.Timestamp.now().month
        rows.append(row)

    seq_df = pd.DataFrame(rows)
    seq_df['SPEED_DELTA'] = seq_df['SPEED'].diff().fillna(0)
    seq_df['SPEED_ROLLING_MEAN'] = seq_df['SPEED'].rolling(window=SEQUENCE_LENGTH, min_periods=1).mean()

    seq_array = seq_df[feature_columns].values
    seq_scaled = scaler.transform(seq_array)
    return seq_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_columns))


def _load_dataframe():
    for path in CANDIDATE_DATASETS:
        if path.exists():
            df = pd.read_csv(path)
            df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']].dropna()
            df = df[df['SPEED'] > 0].copy()
            df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)
            return df
    raise FileNotFoundError("No dataset found.")


@st.cache_resource
def load_data_and_models():
    df = _load_dataframe()

    if LR_MODEL_PATH.exists() and GB_MODEL_PATH.exists():
        lr_model = joblib.load(LR_MODEL_PATH)
        gb_model = joblib.load(GB_MODEL_PATH)
        return df, lr_model, gb_model

    X = df[['HOUR', 'DAY_OF_WEEK']]
    y = df['CONGESTION']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))

    joblib.dump(lr_model, LR_MODEL_PATH)
    joblib.dump(gb_model, GB_MODEL_PATH)

    return df, lr_model, gb_model


df, lr_model, gb_model = load_data_and_models()
lstm_model = load_lstm_model()


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


st.markdown("---")
st.subheader("LSTM Prediction")

lstm_scaler, lstm_feature_columns = load_lstm_assets()

if lstm_model is not None and lstm_scaler is not None and predict_btn:
    sequence = build_lstm_sequence(df, hour, day, lstm_scaler, lstm_feature_columns)
    lstm_prob = float(lstm_model.predict(sequence, verbose=0)[0][0])
    lstm_pred = 1 if lstm_prob >= 0.5 else 0

    st.metric("Congestion Probability (LSTM)", f"{lstm_prob:.2f}")

    if lstm_pred == 1:
        st.error("High congestion predicted (LSTM)")
    else:
        st.success("Low congestion predicted (LSTM)")

    st.caption("LSTM uses a synthetic sequence of the 6 hours leading up to the selected hour, based on historical average speeds.")

elif lstm_model is None:
    st.warning("LSTM model unavailable in this deployment environment.")
elif lstm_scaler is None:
    st.warning("LSTM scaler not found. Re-run lstm_model.py to regenerate lstm_scaler.joblib.")


st.markdown("---")
st.subheader("Congestion Pattern by Hour")

hourly_congestion = df.groupby('HOUR')['CONGESTION'].mean()
st.line_chart(hourly_congestion)


st.markdown("---")
st.subheader("About the Model")

st.write("""
This dashboard predicts traffic congestion based on temporal patterns.

**Logistic Regression** and **Gradient Boosting** predict whether congestion is likely *at the selected hour and day*, using those two features as direct inputs.

**LSTM** predicts whether congestion will occur in the *next timestep*, using a rolling sequence of the 6 most recent speed readings and derived features (speed delta, rolling mean). These models are not directly comparable — LR/GB reflect current-period patterns, while LSTM forecasts the next period.
""")


st.markdown("---")
st.subheader("Dataset Information")

st.write("""
The model is trained using historical traffic data from Chicago.

Congestion is defined based on speed thresholds, but speed is not used as an input feature in the prediction models to ensure meaningful pattern learning.
""")
