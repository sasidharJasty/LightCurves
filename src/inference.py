"""Inference utilities for the trained XGBoost + scaler pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .catalog_pipeline import NUMERIC_FEATURES
from .config import ARTIFACTS_DIR


def load_artifacts() -> Tuple[object, RobustScaler, list[str], list[str]]:
    model = joblib.load(ARTIFACTS_DIR / "xgb_tabular.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler_robust.joblib")
    meta_path = ARTIFACTS_DIR / "tabular_columns.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    feature_cols = metadata.get("feature_columns")
    numeric_cols = metadata.get("numeric_columns")
    if feature_cols is None:
        feature_cols = list(model.get_booster().feature_names)  # type: ignore[attr-defined]
    if numeric_cols is None:
        numeric_cols = [c for c in feature_cols if c in NUMERIC_FEATURES]
    return model, scaler, feature_cols, numeric_cols


def preprocess_dataframe(
    df: pd.DataFrame,
    scaler: RobustScaler,
    feature_columns: list[str],
    numeric_columns: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_proc = df.copy()
    df_proc[NUMERIC_FEATURES] = df_proc[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df_proc["num_numeric"] = df_proc[NUMERIC_FEATURES].notna().sum(axis=1)
    df_proc = df_proc[df_proc["num_numeric"] >= 3]
    df_proc["mission_cat"] = (
        df_proc.get("mission_cat", df_proc.get("mission", "UNKNOWN"))
        .replace(["", "NaN", None, np.nan], "UNKNOWN")
        .astype(str)
        .fillna("UNKNOWN")
    )
    X_raw = df_proc[NUMERIC_FEATURES + ["mission_cat"]].copy()
    X = pd.get_dummies(X_raw, columns=["mission_cat"], dummy_na=True)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    X[numeric_columns] = scaler.transform(X[numeric_columns].fillna(0))
    return X, df_proc


def predict_dispositions(df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, feature_cols, numeric_cols = load_artifacts()
    X, df_clean = preprocess_dataframe(df, scaler, feature_cols, numeric_cols)
    probs = model.predict_proba(X)
    predictions = model.predict(X)
    output = df_clean.copy()
    output["prob_false_positive"] = probs[:, 0]
    output["prob_candidate"] = probs[:, 1]
    output["prob_confirmed"] = probs[:, 2]
    output["prediction"] = predictions
    return output


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Load a dataframe and call predict_dispositions(df) programmatically.")
