"""
Train all classical ML models on the full Chicago dataset and save to disk.

Run this locally before deploying:
    python train_models.py

Outputs:
    lr_model.joblib, gb_model.joblib, rf_model.joblib,
    mlp_model.joblib, svm_model.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_DATASETS = [
    BASE_DIR / "Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv",
    BASE_DIR / "traffic_sample.csv",
]

MODEL_PATHS = {
    "lr":  BASE_DIR / "lr_model.joblib",
    "gb":  BASE_DIR / "gb_model.joblib",
    "rf":  BASE_DIR / "rf_model.joblib",
    "mlp": BASE_DIR / "mlp_model.joblib",
    "svm": BASE_DIR / "svm_model.joblib",
}

FEATURE_COLS = ['HOUR', 'DAY_OF_WEEK', 'MONTH', 'IS_WEEKEND', 'IS_RUSH_HOUR',
                'HOUR_SIN', 'HOUR_COS', 'DAY_SIN', 'DAY_COS']


def engineer_features(df):
    df = df.copy()
    df['IS_WEEKEND']  = (df['DAY_OF_WEEK'] >= 6).astype(int)
    df['IS_RUSH_HOUR'] = df['HOUR'].apply(lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0)
    df['HOUR_SIN']    = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS']    = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['DAY_SIN']     = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DAY_COS']     = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    return df


def load_dataset():
    for path in CANDIDATE_DATASETS:
        if path.exists():
            print(f"Loading dataset: {path.name} ({path.stat().st_size // 1_000_000} MB)")
            return pd.read_csv(path)
    raise FileNotFoundError("No dataset found. Place traffic_sample.csv in the project root.")


def prepare(df):
    needed = ['SPEED', 'HOUR', 'DAY_OF_WEEK']
    if 'MONTH' in df.columns:
        needed.append('MONTH')
    df = df[needed].dropna()
    df = df[df['SPEED'] > 0].copy()
    if 'MONTH' not in df.columns:
        df['MONTH'] = 1  # fallback for sample CSV
    df['CONGESTION'] = np.where(df['SPEED'] < 20, 1, 0)
    df = engineer_features(df)
    return df


def train_and_report(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    print(f"Training {name}...")
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    print(f"{name} Results:")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    return model


def main():
    df = prepare(load_dataset())

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Congestion rate: {df['CONGESTION'].mean():.1%}\n")

    X = df[FEATURE_COLS]
    y = df['CONGESTION']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    weights = compute_sample_weight('balanced', y_train)

    print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows\n")

    lr = train_and_report(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_train, y_train, X_test, y_test,
    )

    gb = train_and_report(
        "Gradient Boosting",
        GradientBoostingClassifier(random_state=42),
        X_train, y_train, X_test, y_test,
        sample_weight=weights,
    )

    rf = train_and_report(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        X_train, y_train, X_test, y_test,
    )

    # MLP does not support class_weight or sample_weight — oversample minority class
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng = np.random.default_rng(42)
    extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
    balanced_idx = np.concatenate([np.arange(len(y_train)), extra])
    rng.shuffle(balanced_idx)
    X_train_bal = X_train.iloc[balanced_idx]
    y_train_bal = y_train.iloc[balanced_idx]

    mlp = train_and_report(
        "MLP Neural Network",
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        X_train_bal, y_train_bal, X_test, y_test,
    )

    # SGDClassifier with modified_huber loss = linear SVM with native probabilities
    svm = train_and_report(
        "SVM (SGD / Linear SVM)",
        SGDClassifier(loss='modified_huber', class_weight='balanced', max_iter=1000, random_state=42),
        X_train, y_train, X_test, y_test,
    )

    for key, model in [("lr", lr), ("gb", gb), ("rf", rf), ("mlp", mlp), ("svm", svm)]:
        joblib.dump(model, MODEL_PATHS[key])
        print(f"Saved: {MODEL_PATHS[key].name}")


if __name__ == "__main__":
    main()
