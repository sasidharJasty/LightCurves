"""SHAP explainability helpers for the XGBoost classifier."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from .tabular_training import TabularSplits


def summarize_shap(model, splits: TabularSplits, class_idx: int = 0, sample_idx: int = 0) -> None:
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(splits.X_test)
    feat_names = getattr(splits.X_test, "columns", None)

    shap.summary_plot(shap_values[:, :, class_idx], splits.X_test)
    shap.summary_plot(shap_values[:, :, class_idx], splits.X_test, plot_type="bar")

    if feat_names is None:
        feat_names = [f"feature_{i}" for i in range(splits.X_test.shape[1])]
    shap_df = pd.DataFrame(shap_values[:, :, class_idx], columns=feat_names)
    X_df = pd.DataFrame(splits.X_test, columns=feat_names)
    corr_matrix = shap_df.corrwith(X_df)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix.to_frame(name="Correlation"),
        annot=True,
        cmap="coolwarm",
        center=0,
    )
    plt.title(f"SHAP/Feature correlation (class {class_idx})")
    plt.tight_layout()
    plt.show()

    row = splits.X_test.iloc[sample_idx] if hasattr(splits.X_test, "iloc") else splits.X_test[sample_idx]
    single_values = shap_values[sample_idx, :, class_idx]
    shap.waterfall_plot(
        shap.Explanation(
            values=single_values,
            base_values=explainer.expected_value[class_idx],
            data=row,
        )
    )
    shap.force_plot(
        base_value=float(explainer.expected_value[class_idx]),
        shap_values=single_values,
        features=row,
        matplotlib=True,
    )
    plt.gcf().set_facecolor("white")
    plt.show()
