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
BATCH_SIZE = 32


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
    """
    Aggregate all segment readings into a city-wide weekly traffic profile.

    Each row in the output represents the average speed across all road
    segments for a given (DAY_OF_WEEK, HOUR) combination, sorted
    chronologically through the week. This gives the LSTM a genuine
    temporal sequence to learn from rather than mixing unrelated segments.
    """
    clean_df = df[["SPEED", "HOUR", "DAY_OF_WEEK"]].dropna()
    clean_df = clean_df[clean_df["SPEED"] > 0].copy()

    profile = (
        clean_df.groupby(["DAY_OF_WEEK", "HOUR"])["SPEED"]
        .mean()
        .reset_index()
        .sort_values(["DAY_OF_WEEK", "HOUR"])
        .reset_index(drop=True)
    )

    profile["SPEED_DELTA"] = profile["SPEED"].diff().fillna(0)
    profile["SPEED_ROLLING_MEAN"] = profile["SPEED"].rolling(window=SEQUENCE_LENGTH, min_periods=1).mean()

    # City-wide averages never drop below 20 mph (individual congested segments
    # get diluted by hundreds of free-flowing ones). Use the 20th percentile of
    # city-wide average speeds as the congestion threshold instead, which
    # preserves a ~20% congestion rate consistent with the per-segment data.
    congestion_threshold = profile["SPEED"].quantile(0.20)
    profile["FUTURE_CONGESTION"] = (profile["SPEED"].shift(-1) < congestion_threshold).astype(int)
    profile = profile.iloc[:-1]

    congestion_rate = profile["FUTURE_CONGESTION"].mean()
    print(f"Weekly profile: {len(profile)} time slots across {profile['DAY_OF_WEEK'].nunique()} days")
    print(f"Congestion threshold: {congestion_threshold:.1f} mph (20th percentile), rate: {congestion_rate:.1%}")
    return profile.reset_index(drop=True), congestion_threshold


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
    df, congestion_threshold = prepare_dataframe(load_dataset())

    feature_columns = [column for column in df.columns if column != "FUTURE_CONGESTION"]
    X = df[feature_columns].values
    y = df["FUTURE_CONGESTION"].values

    split_index = int(len(df) * TRAIN_SPLIT)
    if split_index <= SEQUENCE_LENGTH or len(df) - split_index <= SEQUENCE_LENGTH:
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
    joblib.dump(float(congestion_threshold), base_dir / "lstm_threshold.joblib")
    print("Saved: lstm_model.h5")
    print("Saved: lstm_scaler.joblib")
    print("Saved: lstm_feature_columns.joblib")
    print("Saved: lstm_threshold.joblib")

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
