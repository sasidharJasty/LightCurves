"""Tabular feature engineering, splitting, scaling, and XGBoost training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler

from .catalog_pipeline import LABEL_MAP, NUMERIC_FEATURES, build_canonical_catalog
from .config import ARTIFACTS_DIR, SEED, set_global_seeds


@dataclass
class TabularSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    mission: pd.Series


PARAM_DISTRIBUTIONS: Dict[str, Iterable] = {
    "n_estimators": [100, 300, 600],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df_proc = df.copy()
    df_proc[NUMERIC_FEATURES] = df_proc[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df_proc["mission_cat"] = (
        df_proc.get("mission_cat", df_proc.get("mission", "UNKNOWN"))
        .replace(["", "NaN", None, np.nan], "UNKNOWN")
        .astype(str)
        .fillna("UNKNOWN")
    )
    X_raw = df_proc[NUMERIC_FEATURES + ["mission_cat"]].copy()
    X = pd.get_dummies(X_raw, columns=["mission_cat"], dummy_na=True)
    y = df_proc["label"].astype(int)
    mission = df_proc["mission"].astype(str)
    return X, y, mission


def _mission_based_split(X: pd.DataFrame, y: pd.Series, mission: pd.Series):
    train_mask = mission.str.contains("KEPLER", na=False)
    val_mask = mission.str.contains("K2", na=False)
    test_mask = mission.str.contains("TESS", na=False)

    if train_mask.sum() < 200:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
        )
    else:
        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        X_val = X[val_mask].copy()
        y_val = y[val_mask].copy()
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()
        if len(X_val) < 50 or len(X_test) < 50:
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=SEED
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED
            )
    return TabularSplits(X_train, X_val, X_test, y_train, y_val, y_test, mission)


def compute_numeric_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in NUMERIC_FEATURES]


def scale_splits(splits: TabularSplits) -> Tuple[TabularSplits, RobustScaler, list[str]]:
    scaler = RobustScaler()
    num_cols = compute_numeric_columns(splits.X_train.columns)
    scaler.fit(splits.X_train[num_cols].fillna(0))

    def _scale(df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        df_scaled[num_cols] = scaler.transform(df_scaled[num_cols].fillna(0))
        return df_scaled

    scaled = TabularSplits(
        X_train=_scale(splits.X_train),
        X_val=_scale(splits.X_val),
        X_test=_scale(splits.X_test),
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
        mission=splits.mission,
    )
    return scaled, scaler, num_cols


def train_xgboost(tabular: TabularSplits) -> xgb.XGBClassifier:
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=SEED,
        n_jobs=4,
    )
    search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=18,
        scoring="f1_macro",
        cv=3,
        verbose=2,
        random_state=SEED,
        n_jobs=1,
    )
    search.fit(tabular.X_train, tabular.y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_


def evaluate_model(model: xgb.XGBClassifier, splits: TabularSplits) -> str:
    y_pred_val = model.predict(splits.X_val)
    report = classification_report(splits.y_val, y_pred_val, digits=4)
    print("Validation report:\n", report)
    return report


def persist_artifacts(
    model: xgb.XGBClassifier,
    scaler: RobustScaler,
    feature_columns: list[str],
    numeric_columns: list[str],
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "xgb_tabular.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_robust.joblib")
    payload = {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
    }
    (ARTIFACTS_DIR / "tabular_columns.json").write_text(json.dumps(payload, indent=2))


def run_training_pipeline(force_catalog_download: bool = False) -> Tuple[xgb.XGBClassifier, RobustScaler, TabularSplits]:
    set_global_seeds(SEED)
    df = build_canonical_catalog(force_download=force_catalog_download)
    X, y, mission = prepare_feature_matrix(df)
    splits = _mission_based_split(X, y, mission)
    scaled_splits, scaler, numeric_cols = scale_splits(splits)
    model = train_xgboost(scaled_splits)
    evaluate_model(model, scaled_splits)
    persist_artifacts(model, scaler, list(X.columns), numeric_cols)
    return model, scaler, scaled_splits


if __name__ == "__main__":  # pragma: no cover
    run_training_pipeline()
