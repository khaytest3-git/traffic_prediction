import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# ==============================
# Load dataset
# ==============================

df = pd.read_csv("Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv")

# Keep only needed columns
df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]

print(df.head())
print(df.isnull().sum())

# ==============================
# Data Cleaning
# ==============================

df = df.dropna()
df = df[df['SPEED'] > 0]

print("After cleaning:")
print(df.isnull().sum())

# ==============================
# Create Target Variable
# ==============================

df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)

print(df['CONGESTION'].value_counts())

# ==============================
# Features and Target
# ==============================

X = df[['HOUR', 'DAY_OF_WEEK']]
y = df['CONGESTION']

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Logistic Regression
# ==============================

log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Confusion Matrix - Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)

disp_log = ConfusionMatrixDisplay(
    confusion_matrix=cm_log,
    display_labels=["Not Congested", "Congested"]
)
disp_log.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ==============================
# Random Forest
# ==============================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Not Congested", "Congested"]
)
disp_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

# ==============================
# Model Comparison Table
# ==============================

precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf)
    ],
    'Precision': [precision_log, precision_rf],
    'Recall': [recall_log, recall_rf],
    'F1-Score': [f1_log, f1_rf]
})

print("\nModel Performance Table:")
print(results_df.round(4))

# ==============================
# Feature Importance (Random Forest)
# ==============================

importance = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)