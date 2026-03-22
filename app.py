import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==============================
# Load and prepare data
# ==============================

df = pd.read_csv("traffic_sample.csv")

# Keep needed columns
df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]

# Clean data
df = df.dropna()
df = df[df['SPEED'] > 0]

# Create target
df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)

# Features
X = df[['HOUR', 'DAY_OF_WEEK']]
y = df['CONGESTION']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# ==============================
# Streamlit UI
# ==============================

st.title("Traffic Congestion Prediction")

st.write("This application predicts traffic congestion based on temporal factors such as hour of day and day of week.")

# Create two columns
col1, col2 = st.columns(2)

# ==============================
# LEFT SIDE → Inputs
# ==============================

with col1:
    st.subheader("Input Controls")

    hour = st.slider("Select Hour (0-23)", 0, 23, 8)

    day = st.selectbox(
        "Select Day of Week",
        [1,2,3,4,5,6,7],
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1]
    )

    predict_btn = st.button("Predict")


# ==============================
# RIGHT SIDE → Results
# ==============================

with col2:
    st.subheader("Prediction Output")

    if predict_btn:
        input_data = pd.DataFrame({
            'HOUR': [hour],
            'DAY_OF_WEEK': [day]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.metric("Congestion Probability", f"{probability:.2f}")

        if prediction == 1:
            st.error("High congestion expected")
        else:
            st.success("Low congestion expected")


# ==============================
# Visualization (RQ2)
# ==============================

st.markdown("---")
st.subheader("Congestion Pattern by Hour")

hourly_congestion = df.groupby('HOUR')['CONGESTION'].mean()
st.line_chart(hourly_congestion)


# ==============================
# About Section
# ==============================

st.markdown("---")
st.subheader("About the Model")

st.write("""
This model uses Logistic Regression trained on historical traffic data from Chicago.
The prediction is based on temporal features:
- Hour of day
- Day of week

The model identifies patterns in traffic congestion based on time.
""")
