"""Phase folding utilities built on Lightkurve."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lightkurve import search_lightcurve
from tqdm.auto import tqdm

from .config import ROOT, SEED, set_global_seeds

PHASE_DATA_PATH = ROOT / "phase_folded_data.npz"


@dataclass
class PhaseFoldResult:
    X_phase: np.ndarray | None
    y_phase: np.ndarray | None
    failed: int


def _ensure_period(df: pd.DataFrame) -> pd.DataFrame:
    period_candidates = ["period_days", "period", "pl_orbper", "koi_period"]
    if "period_days" in df.columns:
        return df
    for col in period_candidates:
        if col in df.columns:
            df = df.copy()
            df["period_days"] = df[col]
            return df
    raise KeyError(f"No period column found. Checked: {period_candidates}")


def _ensure_target_id(df: pd.DataFrame) -> pd.DataFrame:
    if "target_id" in df.columns:
        return df
    target_candidates = [
        "kepid",
        "kepoi_name",
        "pl_name",
        "pl_hostname",
        "toi",
        "tic_id",
    ]
    for col in target_candidates:
        if col in df.columns:
            df = df.copy()
            df["target_id"] = df[col]
            return df
    raise KeyError(f"No target identifier found. Checked: {target_candidates}")


def _ensure_label(df: pd.DataFrame) -> pd.DataFrame:
    if "label" in df.columns:
        return df
    label_map = {"FALSE_POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
    df = df.copy()
    df["label"] = df.get("disposition_std", pd.Series([np.nan] * len(df))).map(label_map)
    return df


def make_phase_folded_vector(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float | None = None,
    bins: int = 400,
    fold_window: float = 0.05,
) -> np.ndarray:
    if t0 is None:
        t0 = time[0]
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    mask = np.abs(phase) <= fold_window
    phases = phase[mask]
    fluxes = flux[mask]
    if len(phases) < 10:
        phases = phase
        fluxes = flux
    order = np.argsort(phases)
    phases = phases[order]
    fluxes = fluxes[order]
    bin_edges = np.linspace(-fold_window, fold_window, bins + 1)
    digitized = np.digitize(phases, bin_edges) - 1
    binned = np.zeros(bins)
    counts = np.zeros(bins)
    for idx, bucket in enumerate(digitized):
        if 0 <= bucket < bins:
            binned[bucket] += fluxes[idx]
            counts[bucket] += 1
    counts[counts == 0] = 1
    binned = binned / counts
    binned = (binned - np.nanmedian(binned)) / (np.nanstd(binned) + 1e-9)
    return binned


def phase_fold_sample(
    df: pd.DataFrame,
    sample_size: int = 120,
    bins: int = 400,
    fold_window: float = 0.05,
    save_path: str | None = str(PHASE_DATA_PATH),
) -> PhaseFoldResult:
    set_global_seeds(SEED)
    df = _ensure_label(_ensure_target_id(_ensure_period(df)))
    valid = df[df["period_days"].notna() & df["target_id"].notna()]
    sample = valid.sample(n=min(sample_size, len(valid)), random_state=SEED)

    vectors = []
    labels = []
    failures = 0

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Phase folding"):
        target = row["target_id"]
        period = float(row["period_days"])
        label = int(row.get("label", 1))
        if not np.isfinite(period) or period <= 0:
            failures += 1
            continue
        try:
            query = search_lightcurve(str(target))
            lc_obj = query.download() if query is not None else None
            if lc_obj is None and query is not None:
                bundle = query.download_all()
                lc_obj = bundle[0] if bundle else None
            if lc_obj is None:
                raise ValueError("No light curve found")
            if hasattr(lc_obj, "PDCSAP_FLUX") and lc_obj.PDCSAP_FLUX is not None:
                lc = lc_obj.PDCSAP_FLUX.remove_nans()
            elif hasattr(lc_obj, "SAP_FLUX") and lc_obj.SAP_FLUX is not None:
                lc = lc_obj.SAP_FLUX.remove_nans()
            else:
                lc = getattr(lc_obj, "remove_nans", lambda: lc_obj)()
            if len(lc.time.value) < 5:
                failures += 1
                continue
            vec = make_phase_folded_vector(
                lc.time.value,
                lc.flux.value,
                period,
                bins=bins,
                fold_window=fold_window,
            )
            vectors.append(vec)
            labels.append(label)
        except Exception:
            failures += 1
            continue

    if vectors:
        X_phase = np.vstack(vectors)
        y_phase = np.asarray(labels, dtype=np.int64)
    else:
        X_phase = None
        y_phase = None

    if save_path and X_phase is not None and y_phase is not None:
        np.savez_compressed(save_path, X_phase=X_phase, y_phase=y_phase)
        print(f"Saved phase-folded dataset to {save_path}")
    return PhaseFoldResult(X_phase, y_phase, failures)


def load_phase_data(path: str | None = str(PHASE_DATA_PATH)) -> Tuple[np.ndarray | None, np.ndarray | None]:
    file = Path(path)
    if file.exists():
        data = np.load(file)
        return data.get("X_phase"), data.get("y_phase")
    return None, None


if __name__ == "__main__":  # pragma: no cover
    from .catalog_pipeline import build_canonical_catalog

    df_catalog = build_canonical_catalog(force_download=False)
    phase_fold_sample(df_catalog)
