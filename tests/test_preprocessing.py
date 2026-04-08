"""Tests for data preprocessing logic shared across models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lstm"))

from lstm_model import build_sequences, prepare_dataframe


@pytest.fixture
def raw_df():
    return pd.read_csv(ROOT / "traffic_sample.csv")


def test_csv_loads(raw_df):
    assert not raw_df.empty
    assert "SPEED" in raw_df.columns


def test_prepare_dataframe_drops_nulls(raw_df):
    df = prepare_dataframe(raw_df)
    assert df.isnull().sum().sum() == 0


def test_prepare_dataframe_removes_zero_speed(raw_df):
    df = prepare_dataframe(raw_df)
    assert (df["SPEED"] > 0).all()


def test_prepare_dataframe_adds_features(raw_df):
    df = prepare_dataframe(raw_df)
    assert "SPEED_DELTA" in df.columns
    assert "SPEED_ROLLING_MEAN" in df.columns
    assert "FUTURE_CONGESTION" in df.columns


def test_future_congestion_is_binary(raw_df):
    df = prepare_dataframe(raw_df)
    assert set(df["FUTURE_CONGESTION"].unique()).issubset({0, 1})


def test_build_sequences_shape():
    sequence_length = 6
    n_samples = 50
    n_features = 3
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    X_seq, y_seq = build_sequences(X, y, sequence_length)

    assert X_seq.shape == (n_samples - sequence_length, sequence_length, n_features)
    assert y_seq.shape == (n_samples - sequence_length,)


def test_build_sequences_values():
    sequence_length = 3
    X = np.arange(10).reshape(10, 1).astype(float)
    y = np.zeros(10)

    X_seq, _ = build_sequences(X, y, sequence_length)

    # First sequence should be rows 0,1,2
    np.testing.assert_array_equal(X_seq[0].flatten(), [0, 1, 2])
    # Second sequence should be rows 1,2,3
    np.testing.assert_array_equal(X_seq[1].flatten(), [1, 2, 3])
