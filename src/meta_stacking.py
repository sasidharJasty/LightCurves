"""Meta-learning layer that stacks probabilistic learners."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from .config import ARTIFACTS_DIR
from .tabular_training import TabularSplits


def train_meta_learner(model, splits: TabularSplits) -> LogisticRegression:
    xgb_val_probs = model.predict_proba(splits.X_val)
    xgb_test_probs = model.predict_proba(splits.X_test)

    meta_X = pd.DataFrame(
        xgb_val_probs,
        columns=["xgb_prob_0", "xgb_prob_1", "xgb_prob_2"],
    ).reset_index(drop=True)
    meta_y = splits.y_val.reset_index(drop=True)

    meta_clf = LogisticRegression(multi_class="multinomial", max_iter=1000)
    meta_clf.fit(meta_X, meta_y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_clf, ARTIFACTS_DIR / "meta_logreg.joblib")

    meta_test_X = pd.DataFrame(
        xgb_test_probs,
        columns=["xgb_prob_0", "xgb_prob_1", "xgb_prob_2"],
    )
    meta_preds = meta_clf.predict(meta_test_X)
    report = classification_report(splits.y_test, meta_preds, digits=4)
    print("Stacked model report:\n", report)
    cm = confusion_matrix(splits.y_test, meta_preds)
    print("Confusion matrix:\n", cm)
    return meta_clf
