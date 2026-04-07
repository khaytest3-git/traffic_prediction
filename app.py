import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


def load_lstm_model():
    try:
        from tensorflow.keras.models import load_model
        return load_model("lstm_model.h5")
    except:
        return None


df = pd.read_csv("traffic_sample.csv")

df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]
df = df.dropna()
df = df[df['SPEED'] > 0]

df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)


X = df[['HOUR', 'DAY_OF_WEEK', 'SPEED']]
y = df['CONGESTION']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)


st.title("Traffic Congestion Prediction")

st.write("This application predicts traffic congestion using Logistic Regression, Gradient Boosting, and LSTM models.")


st.subheader("Logistic Regression Prediction")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Select Hour (0-23)", 0, 23, 8)

    speed = st.slider("Select Speed (mph)", 0, 60, 30)

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
            'DAY_OF_WEEK': [day],
            'SPEED': [speed]
        })

        prediction = lr_model.predict(input_data)[0]
        probability = lr_model.predict_proba(input_data)[0][1]

        st.caption("Prediction powered by Logistic Regression")

        st.metric("Congestion Probability", f"{probability:.2f}")

        if prediction == 1:
            st.error("High congestion expected")
        else:
            st.success("Low congestion expected")


st.markdown("---")
st.subheader("Gradient Boosting Prediction")

col1, col2 = st.columns(2)

with col1:
    st.write("Using the same input values:")

with col2:
    if predict_btn:
        input_data = pd.DataFrame({
            'HOUR': [hour],
            'DAY_OF_WEEK': [day],
            'SPEED': [speed]
        })

        gb_prediction = gb_model.predict(input_data)[0]
        gb_probability = gb_model.predict_proba(input_data)[0][1]

        st.caption("Prediction powered by Gradient Boosting")

        st.metric("Congestion Probability (GB)", f"{gb_probability:.2f}")

        if gb_prediction == 1:
            st.error("High congestion expected (GB)")
        else:
            st.success("Low congestion expected (GB)")


lstm_model = load_lstm_model()

st.markdown("---")
st.subheader("LSTM Prediction (Recent Traffic Data)")

if lstm_model is not None:
    df['SPEED_DELTA'] = df['SPEED'].diff().fillna(0)
    df['SPEED_ROLLING_MEAN'] = df['SPEED'].rolling(window=3).mean().fillna(method='bfill')

    features = ['SPEED', 'HOUR', 'DAY_OF_WEEK', 'SPEED_DELTA', 'SPEED_ROLLING_MEAN']

    sequence_data = df[features].values
    latest_sequence = sequence_data[-6:]
    latest_sequence = latest_sequence.reshape(1, 6, len(features))

    lstm_prob = lstm_model.predict(latest_sequence)[0][0]
    lstm_pred = 1 if lstm_prob >= 0.5 else 0

    st.metric("LSTM Congestion Probability", f"{lstm_prob:.2f}")

    if lstm_pred == 1:
        st.error("High congestion predicted (LSTM)")
    else:
        st.success("Low congestion predicted (LSTM)")

    st.caption("LSTM uses recent historical sequence data")

else:
    st.warning("LSTM model unavailable in this deployment environment.")


st.markdown("---")
st.subheader("Congestion Pattern by Hour")

hourly_congestion = df.groupby('HOUR')['CONGESTION'].mean()
st.line_chart(hourly_congestion)


st.markdown("---")
st.subheader("About the Model")

st.write("""
This application compares Logistic Regression, Gradient Boosting, and LSTM models for traffic congestion prediction.

Logistic Regression and Gradient Boosting use temporal and speed features, while LSTM uses sequential traffic data to capture time-based patterns.
""")


st.markdown("---")
st.subheader("Dataset Information")

st.write("""
The model is trained using historical traffic data from Chicago.

A sampled dataset is used in this deployed version to ensure efficient performance.
""")

