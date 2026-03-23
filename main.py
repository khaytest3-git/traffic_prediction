import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv")

# Keep only needed columns
df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]

# Show result
print(df.head())
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Remove invalid speed values
df = df[df['SPEED'] > 0]

print("After cleaning:")
print(df.isnull().sum())

# Create congestion label (1 = congested, 0 = not congested)
df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)

# Check distribution
print(df['CONGESTION'].value_counts())

# Features (IMPORTANT: exclude SPEED to avoid data leakage)
X = df[['HOUR', 'DAY_OF_WEEK']]

# Target
y = df['CONGESTION']

# Create congestion label
df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)

print(df['CONGESTION'].value_counts())


# ==============================
# ADD NEW CODE BELOW THIS LINE
# ==============================

# Step 10: Features and target
X = df[['HOUR', 'DAY_OF_WEEK']]
y = df['CONGESTION']


# Step 11: Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 12: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Step 13: Random Forest Classifier
# ==============================
# Random Forest Model
# ==============================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# Evaluation
print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# ==============================
# Model Comparison
# ==============================

print("\nModel Comparison")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# Feature importance from Random Forest
import pandas as pd

importance = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

print("\nFeature Importance:")
print(importance_df)

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nSorted Feature Importance:")
print(importance_df)