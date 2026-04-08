from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, LSTM, Dense
from tensorflow.keras.models import Sequential


SEQUENCE_LENGTH = 6
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
EPOCHS = 12
BATCH_SIZE = 64
MAX_ROWS = 80000


def load_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    candidate_files = [
        base_dir / "Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current_20260222.csv",
        base_dir / "traffic_sample.csv",
    ]

    for csv_path in candidate_files:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Loaded dataset: {csv_path.name}")
            return df

    raise FileNotFoundError("No traffic dataset was found in the project root.")


def prepare_dataframe(df):
    base_columns = ["SPEED"]
    optional_columns = [column for column in ["HOUR", "DAY_OF_WEEK", "MONTH"] if column in df.columns]
    required_columns = base_columns + optional_columns

    clean_df = df[required_columns].dropna().copy()
    clean_df = clean_df[clean_df["SPEED"] > 0]
    clean_df["SPEED_DELTA"] = clean_df["SPEED"].diff().fillna(0)
    clean_df["SPEED_ROLLING_MEAN"] = clean_df["SPEED"].rolling(window=SEQUENCE_LENGTH, min_periods=1).mean()
    clean_df["FUTURE_CONGESTION"] = (clean_df["SPEED"].shift(-1) < 20).astype(int)
    clean_df = clean_df.iloc[:-1]
    if len(clean_df) > MAX_ROWS:
        clean_df = clean_df.tail(MAX_ROWS)
    return clean_df.reset_index(drop=True)


def build_sequences(feature_array, target_array, sequence_length):
    X_sequences = []
    y_sequences = []

    for index in range(sequence_length, len(feature_array)):
        X_sequences.append(feature_array[index - sequence_length:index])
        y_sequences.append(target_array[index])

    return np.array(X_sequences), np.array(y_sequences)


def compute_class_weights(target_array):
    class_counts = np.bincount(target_array)
    total = len(target_array)
    return {
        label: np.sqrt(total / (len(class_counts) * count))
        for label, count in enumerate(class_counts)
        if count > 0
    }


def oversample_sequences(X_array, y_array, target_positive_ratio=0.28, random_state=42):
    rng = np.random.default_rng(random_state)
    positive_indices = np.where(y_array == 1)[0]
    negative_indices = np.where(y_array == 0)[0]

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return X_array, y_array

    current_positive_ratio = len(positive_indices) / len(y_array)
    if current_positive_ratio >= target_positive_ratio:
        return X_array, y_array

    target_positive_count = int((target_positive_ratio * len(negative_indices)) / (1 - target_positive_ratio))
    additional_count = max(0, target_positive_count - len(positive_indices))
    if additional_count == 0:
        return X_array, y_array

    sampled_positive_indices = rng.choice(positive_indices, size=additional_count, replace=True)
    combined_indices = np.concatenate([negative_indices, positive_indices, sampled_positive_indices])
    rng.shuffle(combined_indices)
    return X_array[combined_indices], y_array[combined_indices]


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_score = -1.0
    best_accuracy = -1.0

    for threshold in np.arange(0.35, 0.71, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        if score > best_score or (np.isclose(score, best_score) and accuracy > best_accuracy):
            best_score = score
            best_threshold = float(threshold)
            best_accuracy = accuracy

    return best_threshold, best_score


def main():
    df = prepare_dataframe(load_dataset())

    feature_columns = [column for column in df.columns if column != "FUTURE_CONGESTION"]
    X = df[feature_columns].values
    y = df["FUTURE_CONGESTION"].values

    split_index = int(len(df) * TRAIN_SPLIT)
    if split_index <= SEQUENCE_LENGTH or len(df) - split_index <= 1:
        raise ValueError("Dataset is too small for the configured sequence length and train/test split.")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X[:split_index])
    X_test_scaled = scaler.transform(X[split_index - SEQUENCE_LENGTH:])

    y_train = y[:split_index]
    y_test = y[split_index - SEQUENCE_LENGTH:]

    X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = build_sequences(X_test_scaled, y_test, SEQUENCE_LENGTH)

    validation_size = max(1, int(len(X_train_seq) * VALIDATION_SPLIT))
    if validation_size >= len(X_train_seq):
        raise ValueError("Training data is too small for the configured validation split.")

    X_train_final = X_train_seq[:-validation_size]
    y_train_final = y_train_seq[:-validation_size]
    X_val = X_train_seq[-validation_size:]
    y_val = y_train_seq[-validation_size:]

    X_train_balanced, y_train_balanced = oversample_sequences(X_train_final, y_train_final)
    class_weights = compute_class_weights(y_train_final)
    class_weights[1] = min(class_weights.get(1, 1.0), 1.5)

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, len(feature_columns))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    model.fit(
        X_train_balanced,
        y_train_balanced,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2,
    )

    val_prob = model.predict(X_val, verbose=0).ravel()
    best_threshold, best_f1 = find_best_threshold(y_val, val_prob)

    y_prob = model.predict(X_test_seq, verbose=0).ravel()
    y_pred = (y_prob >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_seq,
        y_pred,
        labels=[1],
        zero_division=0,
    )

    base_dir = Path(__file__).resolve().parent.parent
    model.save(base_dir / "lstm_model.h5")
    joblib.dump(scaler, base_dir / "lstm_scaler.joblib")
    joblib.dump(feature_columns, base_dir / "lstm_feature_columns.joblib")
    print("Saved: lstm_model.h5")
    print("Saved: lstm_scaler.joblib")
    print("Saved: lstm_feature_columns.joblib")

    print("\nLSTM Results")
    print("Best threshold:", best_threshold)
    print("Validation F1:", round(best_f1, 4))
    print("Accuracy:", accuracy_score(y_test_seq, y_pred))
    print("Congestion Precision:", round(float(precision[0]), 4))
    print("Congestion Recall:", round(float(recall[0]), 4))
    print("Congestion F1:", round(float(f1[0]), 4))
    print(classification_report(y_test_seq, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
