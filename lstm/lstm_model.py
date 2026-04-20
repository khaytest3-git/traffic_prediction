"""
Train LSTM and GRU models on per-segment sequences from the Chicago Traffic dataset.

Run this locally before deploying:
    python lstm/lstm_model.py

Outputs (saved to project root):
    lstm_model.h5, gru_model.h5
    lstm_scaler.joblib, lstm_feature_columns.joblib
    lstm_threshold.joblib, lstm_best_threshold.joblib, gru_threshold.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, LSTM, GRU, Dense
from tensorflow.keras.models import Sequential


SEQUENCE_LENGTH  = 6
TRAIN_SPLIT      = 0.8
VALIDATION_SPLIT = 0.1
EPOCHS           = 20
BATCH_SIZE       = 128
CONGESTION_SPEED = 20.0

FEATURE_COLS = ['SPEED', 'HOUR', 'DAY_OF_WEEK', 'SPEED_DELTA', 'SPEED_ROLLING_MEAN']


def load_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv",
        base_dir / "traffic_sample.csv",
    ]
    for path in candidates:
        if path.exists():
            print(f"Loaded dataset: {path.name} ({path.stat().st_size // 1_000_000} MB)")
            return pd.read_csv(path)
    raise FileNotFoundError("No traffic dataset found.")


def prepare_segments(df):
    """Build per-segment feature/target arrays sorted chronologically."""
    df = df[['SEGMENT_ID', 'TIME', 'SPEED', 'HOUR', 'DAY_OF_WEEK']].dropna()
    df = df[df['SPEED'] > 0].copy()

    segments = []
    for _, group in df.groupby('SEGMENT_ID'):
        group = group.sort_values('TIME').reset_index(drop=True)
        if len(group) < SEQUENCE_LENGTH + 2:
            continue
        group['SPEED_DELTA']        = group['SPEED'].diff().fillna(0)
        group['SPEED_ROLLING_MEAN'] = group['SPEED'].rolling(window=SEQUENCE_LENGTH, min_periods=1).mean()
        group['CONGESTION']         = (group['SPEED'] < CONGESTION_SPEED).astype(int)
        split = int(len(group) * TRAIN_SPLIT)
        segments.append({
            'features': group[FEATURE_COLS].values.astype(np.float32),
            'targets':  group['CONGESTION'].values,
            'split':    split,
        })
    return segments


def build_sequences_and_scaler(segments):
    """Fit scaler on training rows only, then build train/test sequences."""
    train_rows = np.vstack([s['features'][:s['split']] for s in segments])
    scaler = MinMaxScaler()
    scaler.fit(train_rows)

    X_train, y_train, X_test, y_test = [], [], [], []

    for s in segments:
        scaled  = scaler.transform(s['features'])
        targets = s['targets']
        split   = s['split']
        n       = len(scaled)

        for i in range(SEQUENCE_LENGTH, split):
            X_train.append(scaled[i - SEQUENCE_LENGTH:i])
            y_train.append(targets[i])

        for i in range(split, n):
            X_test.append(scaled[i - SEQUENCE_LENGTH:i])
            y_test.append(targets[i])

    return (
        np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32),
        np.array(X_test,  dtype=np.float32), np.array(y_test,  dtype=np.int32),
        scaler,
    )


def compute_class_weights(y):
    counts = np.bincount(y)
    total  = len(y)
    return {
        label: np.sqrt(total / (len(counts) * count))
        for label, count in enumerate(counts) if count > 0
    }


def oversample_sequences(X, y, target_positive_ratio=0.28, random_state=42):
    rng = np.random.default_rng(random_state)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0 or len(pos) / len(y) >= target_positive_ratio:
        return X, y
    target_pos  = int(target_positive_ratio * len(neg) / (1 - target_positive_ratio))
    extra       = rng.choice(pos, size=max(0, target_pos - len(pos)), replace=True)
    idx         = np.concatenate([neg, pos, extra])
    rng.shuffle(idx)
    return X[idx], y[idx]


def find_best_threshold(y_true, y_prob):
    best_t, best_f1, best_acc = 0.5, -1.0, -1.0
    for t in np.arange(0.30, 0.71, 0.05):
        pred    = (y_prob >= t).astype(int)
        score   = f1_score(y_true, pred, zero_division=0)
        acc     = accuracy_score(y_true, pred)
        if score > best_f1 or (np.isclose(score, best_f1) and acc > best_acc):
            best_t, best_f1, best_acc = float(t), score, acc
    return best_t, best_f1


def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=2,
    )
    return model


def evaluate(name, model, X_val, y_val, X_test, y_test):
    val_prob  = model.predict(X_val,  verbose=0).ravel()
    threshold, val_f1 = find_best_threshold(y_val, val_prob)

    test_prob = model.predict(X_test, verbose=0).ravel()
    test_pred = (test_prob >= threshold).astype(int)

    p, r, f, _ = precision_recall_fscore_support(y_test, test_pred, labels=[1], zero_division=0)
    print(f"\n{name} Results  (threshold={threshold:.2f})")
    print(f"  Val F1: {val_f1:.4f}")
    print(f"  Test Accuracy : {accuracy_score(y_test, test_pred):.4f}")
    print(f"  Congestion  P : {float(p[0]):.4f}  R : {float(r[0]):.4f}  F1 : {float(f[0]):.4f}")
    print(classification_report(y_test, test_pred, zero_division=0))
    return threshold


def main():
    df = load_dataset()

    print("\nPreparing per-segment sequences...")
    segments = prepare_segments(df)
    print(f"Usable segments : {len(segments)}")

    X_train, y_train, X_test, y_test, scaler = build_sequences_and_scaler(segments)
    print(f"Train sequences : {len(X_train):,}")
    print(f"Test  sequences : {len(X_test):,}")
    print(f"Congestion rate : {y_train.mean():.1%}\n")

    val_size      = max(1, int(len(X_train) * VALIDATION_SPLIT))
    X_val         = X_train[-val_size:]
    y_val         = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    X_bal, y_bal  = oversample_sequences(X_train_final, y_train_final)
    cw            = compute_class_weights(y_train_final)
    cw[1]         = min(cw.get(1, 1.0), 1.5)

    n_features = len(FEATURE_COLS)
    base_dir   = Path(__file__).resolve().parent.parent

    # --- LSTM ---
    print("Training LSTM...")
    lstm = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    lstm = train_model(lstm, X_bal, y_bal, X_val, y_val, cw)
    lstm_threshold = evaluate('LSTM', lstm, X_val, y_val, X_test, y_test)

    lstm.save(base_dir / 'lstm_model.h5')
    joblib.dump(scaler,       base_dir / 'lstm_scaler.joblib')
    joblib.dump(FEATURE_COLS, base_dir / 'lstm_feature_columns.joblib')
    joblib.dump(CONGESTION_SPEED,  base_dir / 'lstm_threshold.joblib')
    joblib.dump(lstm_threshold,    base_dir / 'lstm_best_threshold.joblib')
    print("Saved: lstm_model.h5  lstm_scaler.joblib  lstm_best_threshold.joblib")

    # --- GRU ---
    print("\nTraining GRU...")
    gru = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    gru = train_model(gru, X_bal, y_bal, X_val, y_val, cw)
    gru_threshold = evaluate('GRU', gru, X_val, y_val, X_test, y_test)

    gru.save(base_dir / 'gru_model.h5')
    joblib.dump(gru_threshold, base_dir / 'gru_threshold.joblib')
    print("Saved: gru_model.h5  gru_threshold.joblib")


if __name__ == '__main__':
    main()
