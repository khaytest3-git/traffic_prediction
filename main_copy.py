import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load dataset
df = pd.read_csv("Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv")

# Take a smaller sample to speed up training (adjust sample size as needed)
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

# Keep only needed columns
df = df[['SPEED', 'HOUR', 'DAY_OF_WEEK']]

print(df.head())
print(df.isnull().sum())

# Data Cleaning

df = df.dropna()
df = df[df['SPEED'] > 0]

print("After cleaning:")
print(df.isnull().sum())

# Create Target Variable
df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)

print(df['CONGESTION'].value_counts())


# Features and Target
X = df[['HOUR', 'DAY_OF_WEEK']]
y = df['CONGESTION']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Logistic Regression
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

# Save the plot
plt.savefig("logistic_regression_confusion_matrix.png")
print("Confusion matrix saved as 'logistic_regression_confusion_matrix.png'")

# Try to show the plot (may not work in terminal environments)
try:
    plt.show()
except:
    print("Note: Plot display may not work in terminal. Check the saved PNG file.")

# Early exit - stop here
exit()