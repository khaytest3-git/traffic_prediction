"""Tests for LR and GB model training and prediction."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def trained_models():
    rng = np.random.default_rng(42)
    hours = rng.integers(0, 24, 500)
    days = rng.integers(1, 8, 500)
    speeds = rng.uniform(5, 60, 500)
    congestion = (speeds < 20).astype(int)

    df = pd.DataFrame({"HOUR": hours, "DAY_OF_WEEK": days, "CONGESTION": congestion})
    X = df[["HOUR", "DAY_OF_WEEK"]]
    y = df["CONGESTION"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)

    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)

    return lr, gb, X_test


def test_lr_predict_shape(trained_models):
    lr, _, X_test = trained_models
    preds = lr.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_gb_predict_shape(trained_models):
    _, gb, X_test = trained_models
    preds = gb.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_lr_predict_proba_bounds(trained_models):
    lr, _, X_test = trained_models
    proba = lr.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_gb_predict_proba_bounds(trained_models):
    _, gb, X_test = trained_models
    proba = gb.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_lr_binary_output(trained_models):
    lr, _, X_test = trained_models
    preds = lr.predict(X_test)
    assert set(preds).issubset({0, 1})


def test_gb_binary_output(trained_models):
    _, gb, X_test = trained_models
    preds = gb.predict(X_test)
    assert set(preds).issubset({0, 1})


def test_single_input_prediction(trained_models):
    lr, gb, _ = trained_models
    single = pd.DataFrame({"HOUR": [8], "DAY_OF_WEEK": [2]})
    assert lr.predict(single)[0] in {0, 1}
    assert gb.predict(single)[0] in {0, 1}
