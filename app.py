from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent

FEATURE_COLS = ['HOUR', 'DAY_OF_WEEK', 'MONTH', 'IS_WEEKEND', 'IS_RUSH_HOUR',
                'HOUR_SIN', 'HOUR_COS', 'DAY_SIN', 'DAY_COS']


def engineer_features(hour, day, month):
    is_weekend  = int(day >= 6)
    is_rush     = int(7 <= hour <= 9 or 16 <= hour <= 19)
    hour_sin    = np.sin(2 * np.pi * hour / 24)
    hour_cos    = np.cos(2 * np.pi * hour / 24)
    day_sin     = np.sin(2 * np.pi * day / 7)
    day_cos     = np.cos(2 * np.pi * day / 7)
    return pd.DataFrame([{
        'HOUR': hour, 'DAY_OF_WEEK': day, 'MONTH': month,
        'IS_WEEKEND': is_weekend, 'IS_RUSH_HOUR': is_rush,
        'HOUR_SIN': hour_sin, 'HOUR_COS': hour_cos,
        'DAY_SIN': day_sin, 'DAY_COS': day_cos,
    }])[FEATURE_COLS]
LOOKUP_PATH = BASE_DIR / "chicago_hourly_lookup.csv"
LSTM_MODEL_PATH = BASE_DIR / "lstm_model.h5"
LSTM_SCALER_PATH = BASE_DIR / "lstm_scaler.joblib"
LSTM_FEATURES_PATH = BASE_DIR / "lstm_feature_columns.joblib"
LSTM_THRESHOLD_PATH      = BASE_DIR / "lstm_threshold.joblib"
LSTM_BEST_THRESHOLD_PATH = BASE_DIR / "lstm_best_threshold.joblib"
GRU_MODEL_PATH           = BASE_DIR / "gru_model.h5"
GRU_THRESHOLD_PATH       = BASE_DIR / "gru_threshold.joblib"
LR_MODEL_PATH  = BASE_DIR / "lr_model.joblib"
GB_MODEL_PATH  = BASE_DIR / "gb_model.joblib"
RF_MODEL_PATH  = BASE_DIR / "rf_model.joblib"
MLP_MODEL_PATH = BASE_DIR / "mlp_model.joblib"
SVM_MODEL_PATH = BASE_DIR / "svm_model.joblib"
SEQUENCE_LENGTH = 6


@st.cache_resource
def load_lookup():
    return pd.read_csv(LOOKUP_PATH)


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
def load_gru_model():
    try:
        from tensorflow.keras.models import load_model
        if not GRU_MODEL_PATH.exists():
            return None
        return load_model(GRU_MODEL_PATH)
    except Exception:
        return None


@st.cache_resource
def load_lstm_assets():
    if not LSTM_SCALER_PATH.exists() or not LSTM_FEATURES_PATH.exists():
        return None, None, None
    scaler = joblib.load(LSTM_SCALER_PATH)
    feature_columns = joblib.load(LSTM_FEATURES_PATH)
    threshold = joblib.load(LSTM_THRESHOLD_PATH) if LSTM_THRESHOLD_PATH.exists() else 20.0
    return scaler, feature_columns, threshold


def build_lstm_sequence(lookup, hour, day, scaler, feature_columns):
    speed_lookup = lookup.set_index(['HOUR', 'DAY_OF_WEEK'])['AVG_SPEED']
    global_mean_speed = lookup['AVG_SPEED'].mean()

    rows = []
    for i in range(SEQUENCE_LENGTH):
        h = (hour - SEQUENCE_LENGTH + 1 + i) % 24
        speed = speed_lookup.get((h, day), global_mean_speed)
        row = {'SPEED': speed, 'HOUR': h, 'DAY_OF_WEEK': day}
        rows.append(row)

    seq_df = pd.DataFrame(rows)
    seq_df['SPEED_DELTA'] = seq_df['SPEED'].diff().fillna(0)
    seq_df['SPEED_ROLLING_MEAN'] = seq_df['SPEED'].rolling(window=SEQUENCE_LENGTH, min_periods=1).mean()

    seq_array = seq_df[feature_columns].values
    seq_scaled = scaler.transform(seq_array)
    return seq_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_columns))


@st.cache_resource
def load_models():
    paths = [LR_MODEL_PATH, GB_MODEL_PATH, RF_MODEL_PATH, MLP_MODEL_PATH, SVM_MODEL_PATH]
    if all(p.exists() for p in paths):
        return tuple(joblib.load(p) for p in paths)

    # Fallback: train from traffic_sample.csv if joblib files are missing
    sample_path = BASE_DIR / "traffic_sample.csv"
    df = pd.read_csv(sample_path)
    df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']].dropna()
    df = df[df['SPEED'] > 0].copy()
    df['MONTH'] = 1
    df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)
    df['IS_WEEKEND']   = (df['DAY_OF_WEEK'] >= 6).astype(int)
    df['IS_RUSH_HOUR'] = df['HOUR'].apply(lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0)
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['DAY_SIN']  = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DAY_COS']  = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

    X = df[FEATURE_COLS]
    y = df['CONGESTION']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    weights = compute_sample_weight('balanced', y_train)

    lr  = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)

    gb  = GradientBoostingClassifier()
    gb.fit(X_train, y_train, sample_weight=weights)

    rf  = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    pos_idx = np.where(y_train.values == 1)[0]
    neg_idx = np.where(y_train.values == 0)[0]
    rng = np.random.default_rng(42)
    extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
    bal_idx = np.concatenate([np.arange(len(y_train)), extra])
    rng.shuffle(bal_idx)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(X_train.iloc[bal_idx], y_train.iloc[bal_idx])

    svm = SGDClassifier(loss='modified_huber', class_weight='balanced', max_iter=1000, random_state=42)
    svm.fit(X_train, y_train)

    for model, path in zip([lr, gb, rf, mlp, svm], paths):
        joblib.dump(model, path)

    return lr, gb, rf, mlp, svm


lookup = load_lookup()
lr_model, gb_model, rf_model, mlp_model, svm_model = load_models()
lstm_model = load_lstm_model()
gru_model_seq = load_gru_model()


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

    month = st.selectbox(
        "Select Month",
        list(range(1, 13)),
        index=0,
        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1]
    )

    predict_btn = st.button("Predict")

with col2:
    if predict_btn:
        input_data = engineer_features(hour, day, month)

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
    input_data = engineer_features(hour, day, month)

    gb_prediction = gb_model.predict(input_data)[0]
    gb_probability = gb_model.predict_proba(input_data)[0][1]

    st.caption("Gradient Boosting")
    st.metric("Congestion Probability (GB)", f"{gb_probability:.2f}")

    if gb_prediction == 1:
        st.error("High congestion expected (GB)")
    else:
        st.success("Low congestion expected (GB)")


st.markdown("---")
st.subheader("Random Forest Prediction")

if predict_btn:
    input_data = engineer_features(hour, day, month)
    rf_prediction = rf_model.predict(input_data)[0]
    rf_probability = rf_model.predict_proba(input_data)[0][1]

    st.caption("Random Forest")
    st.metric("Congestion Probability (RF)", f"{rf_probability:.2f}")
    if rf_prediction == 1:
        st.error("High congestion expected (RF)")
    else:
        st.success("Low congestion expected (RF)")


st.markdown("---")
st.subheader("MLP Neural Network Prediction")

if predict_btn:
    input_data = engineer_features(hour, day, month)
    mlp_prediction = mlp_model.predict(input_data)[0]
    mlp_probability = mlp_model.predict_proba(input_data)[0][1]

    st.caption("MLP Neural Network")
    st.metric("Congestion Probability (MLP)", f"{mlp_probability:.2f}")
    if mlp_prediction == 1:
        st.error("High congestion expected (MLP)")
    else:
        st.success("Low congestion expected (MLP)")


st.markdown("---")
st.subheader("SVM Prediction")

if predict_btn:
    input_data = engineer_features(hour, day, month)
    svm_prediction = svm_model.predict(input_data)[0]
    svm_probability = svm_model.predict_proba(input_data)[0][1]

    st.caption("Support Vector Machine (Linear)")
    st.metric("Congestion Probability (SVM)", f"{svm_probability:.2f}")
    if svm_prediction == 1:
        st.error("High congestion expected (SVM)")
    else:
        st.success("Low congestion expected (SVM)")


st.markdown("---")
st.subheader("LSTM Prediction")

lstm_scaler, lstm_feature_columns, lstm_threshold = load_lstm_assets()

if lstm_model is not None and lstm_scaler is not None and predict_btn:
    sequence = build_lstm_sequence(lookup, hour, day, lstm_scaler, lstm_feature_columns)
    lstm_prob = float(lstm_model.predict(sequence, verbose=0)[0][0])
    lstm_best_t = joblib.load(LSTM_BEST_THRESHOLD_PATH) if LSTM_BEST_THRESHOLD_PATH.exists() else 0.5
    lstm_pred = 1 if lstm_prob >= lstm_best_t else 0

    st.metric("Congestion Probability (LSTM)", f"{lstm_prob:.2f}")

    if lstm_pred == 1:
        st.error("Below-average city speed predicted (LSTM)")
    else:
        st.success("Normal city speed predicted (LSTM)")

    if lstm_threshold is not None:
        st.caption(f"LSTM predicts whether the next hour's city-wide average speed will fall below {lstm_threshold:.1f} mph (slowest 20% of hours). Uses the 6 hours leading up to the selected time.")

elif lstm_model is None:
    st.warning("LSTM model unavailable in this deployment environment.")
elif lstm_scaler is None:
    st.warning("LSTM scaler not found. Re-run lstm_model.py to regenerate lstm_scaler.joblib.")


st.markdown("---")
st.subheader("GRU Prediction")

if gru_model_seq is not None and lstm_scaler is not None and predict_btn:
    gru_sequence = build_lstm_sequence(lookup, hour, day, lstm_scaler, lstm_feature_columns)
    gru_prob = float(gru_model_seq.predict(gru_sequence, verbose=0)[0][0])
    gru_threshold = joblib.load(GRU_THRESHOLD_PATH) if GRU_THRESHOLD_PATH.exists() else 0.5
    gru_pred = 1 if gru_prob >= gru_threshold else 0

    st.metric("Congestion Probability (GRU)", f"{gru_prob:.2f}")

    if gru_pred == 1:
        st.error("Below-average city speed predicted (GRU)")
    else:
        st.success("Normal city speed predicted (GRU)")

    st.caption(f"GRU uses the same 6-hour sequence as LSTM but with a single GRU layer (32 units) — fewer parameters, comparable accuracy.")

elif gru_model_seq is None:
    st.warning("GRU model not found. Run lstm/lstm_model.py to generate gru_model.h5.")


st.markdown("---")
st.subheader("Congestion Pattern by Hour")

hourly_congestion = lookup.groupby('HOUR')['CONGESTION_RATE'].mean()
st.line_chart(hourly_congestion)


st.markdown("---")
st.subheader("About the Models")

st.write("""
All models predict the likelihood of traffic congestion based on time patterns learned from historical Chicago traffic data. Congestion is defined as **speed < 20 mph**.

**Logistic Regression**
A linear classifier that finds the best decision boundary between congested and non-congested hours. Trained with balanced class weights. Tends to give higher probabilities due to this balancing.

**Gradient Boosting**
An ensemble of decision trees trained sequentially, each correcting the errors of the previous. Uses balanced sample weights. Generally more conservative than Logistic Regression.

**Random Forest**
An ensemble of independently trained decision trees that vote on the final prediction. Uses balanced class weights and benefits from averaging across 100 trees, reducing overfitting.

**MLP Neural Network**
A feedforward neural network with two hidden layers (64 and 32 neurons). Captures non-linear relationships between time features and congestion. Uses balanced sample weights during training.

**SVM (Support Vector Machine)**
Finds the maximum-margin decision boundary between congested and non-congested classes. Uses a linear kernel via SGDClassifier (stochastic gradient descent with modified Huber loss), which scales efficiently to large datasets and supports native probability estimates and class balancing.

**GRU (Gated Recurrent Unit)**
A lighter alternative to LSTM. Uses the same 6-hour city-wide sequence but replaces the two LSTM layers with a single GRU layer (32 units). GRU combines the forget and input gates into a single update gate, reducing the number of parameters while retaining the ability to capture temporal dependencies. Faster to train and often matches LSTM accuracy on short sequences.

**LSTM (Long Short-Term Memory)**
Trained on a city-wide weekly traffic profile — the average speed across all Chicago road segments for each hour of the week. This gives the LSTM a genuine temporal sequence to learn from (how Monday morning rush connects to the afternoon, how Friday evening transitions, etc.) rather than mixing unrelated road segments.

At prediction time, a 6-hour sequence is built using historical average speeds for the hours leading up to the selected time, scaled to match training, and passed to the model. The LSTM predicts whether the *next* hour's city-wide average speed will be in the slowest 20% of hours.

Because city-wide averages rarely drop below the 20 mph per-segment threshold, the LSTM uses a relative congestion definition (slowest 20th percentile) rather than the fixed threshold used by LR and GB. Their probabilities reflect different scales and are not directly comparable.
""")


st.markdown("---")
st.subheader("Dataset Information")

st.write("""
**Source:** Chicago Traffic Tracker — Historical Congestion Estimates by Segment (2024–present), provided by the City of Chicago.

**Training data:** 508,166 records after cleaning (speed > 0, no nulls), covering road segments across Chicago with speed readings recorded at regular intervals.

**Features used:**
- *Hour of day* (0–23)
- *Day of week* (1 = Monday, 7 = Sunday)
- *Speed* (used by LSTM only, via historical averages)
- *Speed delta* and *rolling mean* (LSTM derived features)

**Class balance:** ~19.7% of records are congested. All models apply balancing techniques to avoid defaulting to always predicting "no congestion".
""")
