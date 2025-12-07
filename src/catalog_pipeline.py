"""Data acquisition and canonical catalog construction pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, SEED, set_global_seeds

CATALOG_SPECS: Dict[str, Tuple[str, Path]] = {
    "kepler_koi": ("select * from q1_q17_dr25_koi", DATA_DIR / "kepler_koi.csv"),
    "pscomppars": ("select * from pscomppars", DATA_DIR / "pscomppars.csv"),
    "tess_toi": ("select * from toi", DATA_DIR / "tess_toi.csv"),
}

CANONICAL_COLUMNS = [
    "mission",
    "target_id",
    "ra",
    "dec",
    "period_days",
    "transit_duration_hours",
    "depth_ppm",
    "planet_radius_re",
    "stellar_radius_rs",
    "stellar_teff_k",
    "stellar_logg",
    "snr",
    "disposition",
    "source",
]

NUMERIC_FEATURES = [
    "period_days",
    "transit_duration_hours",
    "depth_ppm",
    "planet_radius_re",
    "stellar_radius_rs",
    "stellar_teff_k",
    "stellar_logg",
    "snr",
    "log_period",
    "log_radius",
    "depth_over_duration",
]

LABEL_MAP = {
    "CONFIRMED": 2,
    "CANDIDATE": 1,
    "FALSE_POSITIVE": 0,
    "AMBIGUOUS": 1,
}


def download_catalogs(force: bool = False, timeout: int = 60) -> None:
    """Download each catalog if missing or when force=True."""
    session = requests.Session()
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    for name, (query, path) in CATALOG_SPECS.items():
        if path.exists() and not force:
            continue
        q = quote_plus(query).replace("%2A", "*")
        url = f"{base}?request=doQuery&lang=ADQL&format=csv&query={q}"
        print(f"Downloading {name} -> {path}")
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        path.write_bytes(resp.content)


def unify_koi(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["mission"] = "KEPLER"
    out["target_id"] = df.get("kepid", df.get("kepoi_name", np.nan))
    out["ra"] = df.get("ra", np.nan)
    out["dec"] = df.get("dec", np.nan)
    out["period_days"] = df.get("koi_period", np.nan)
    out["transit_duration_hours"] = df.get("koi_duration", np.nan)
    out["depth_ppm"] = df.get("koi_depth", np.nan)
    out["planet_radius_re"] = df.get("koi_prad", np.nan)
    out["stellar_radius_rs"] = df.get("koi_srad", np.nan)
    out["stellar_teff_k"] = df.get("koi_steff", np.nan)
    out["stellar_logg"] = df.get("koi_slogg", np.nan)
    out["snr"] = df.get("koi_model_snr", df.get("koi_prad_err1", np.nan))
    out["disposition"] = df.get("koi_disposition", np.nan)
    out["source"] = "kepler_koi"
    return out


def unify_pscomppars(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    names = df.get("pl_name", pd.Series([np.nan] * len(df)))

    def detect_mission(name: object) -> object:
        if isinstance(name, str):
            if name.startswith("K2-"):
                return "K2"
            if name.startswith("Kepler-") or name.startswith("KOI-"):
                return "KEPLER"
            if name.startswith("TOI"):
                return "TESS"
        return df.get("pl_mission", np.nan)

    out["mission"] = names.apply(detect_mission)
    out["target_id"] = df.get("pl_name", df.get("pl_hostname", np.nan))
    out["ra"] = df.get("ra", np.nan)
    out["dec"] = df.get("dec", np.nan)
    out["period_days"] = df.get("pl_orbper", df.get("pl_orbper_err1", np.nan))
    out["transit_duration_hours"] = df.get("pl_trandur", np.nan)
    out["depth_ppm"] = df.get("pl_trandep", np.nan)
    out["planet_radius_re"] = df.get("pl_rade", df.get("pl_radj", np.nan))
    out["stellar_radius_rs"] = df.get("st_rad", np.nan)
    out["stellar_teff_k"] = df.get("st_teff", np.nan)
    out["stellar_logg"] = df.get("st_logg", np.nan)
    out["snr"] = df.get("pl_rvflag", np.nan)
    out["disposition"] = df.get("pl_discmethod", np.nan)
    out["source"] = "pscomppars"
    return out


def unify_toi(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["mission"] = "TESS"
    out["target_id"] = df.get("toi", df.get("tic_id", np.nan))
    out["ra"] = df.get("ra", np.nan)
    out["dec"] = df.get("dec", np.nan)
    out["period_days"] = df.get("period", df.get("orbital_period", np.nan))
    out["transit_duration_hours"] = df.get("duration_hours", df.get("duration", np.nan))
    out["depth_ppm"] = df.get("depth", np.nan)
    out["planet_radius_re"] = df.get("planet_radius", df.get("radius", np.nan))
    out["stellar_radius_rs"] = df.get("stellar_radius", np.nan)
    out["stellar_teff_k"] = df.get("stellar_teff", np.nan)
    out["stellar_logg"] = df.get("stellar_logg", np.nan)
    out["snr"] = df.get("snr", np.nan)
    out["disposition"] = df.get("tfopwg_disposition", df.get("disposition", np.nan))
    out["source"] = "tess_toi"
    return out


def map_disposition(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    token = str(value).lower()
    if "conf" in token or "known" in token:
        return "CONFIRMED"
    if "cand" in token or "pc" in token:
        return "CANDIDATE"
    if "false" in token or "fp" in token:
        return "FALSE_POSITIVE"
    if "ambig" in token or "apc" in token:
        return "AMBIGUOUS"
    return np.nan


def normalize_depth(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        val = float(value)
        return val * 1e6 if val <= 1.0 else val
    except Exception:
        return np.nan


def normalize_duration(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        val = float(value)
        return val * 24.0 if val < 2 else val
    except Exception:
        return np.nan


def build_canonical_catalog(force_download: bool = False, min_numeric: int = 3) -> pd.DataFrame:
    """Download, unify, and engineer canonical exoplanet features."""
    set_global_seeds(SEED)
    download_catalogs(force=force_download)

    koi_raw = pd.read_csv(CATALOG_SPECS["kepler_koi"][1], low_memory=False)
    ps_raw = pd.read_csv(CATALOG_SPECS["pscomppars"][1], low_memory=False)
    toi_raw = pd.read_csv(CATALOG_SPECS["tess_toi"][1], low_memory=False)

    combined = pd.concat(
        [unify_koi(koi_raw), unify_pscomppars(ps_raw), unify_toi(toi_raw)],
        ignore_index=True,
        sort=False,
    )
    combined["disposition_std"] = combined["disposition"].apply(map_disposition)

    df = combined[combined["disposition_std"].notna()].copy()
    df = df[(df["period_days"].notna()) | (df["planet_radius_re"].notna())]
    df["depth_ppm"] = df["depth_ppm"].apply(normalize_depth)
    df["transit_duration_hours"] = df["transit_duration_hours"].apply(normalize_duration)
    df["log_period"] = np.log10(df["period_days"].replace(0, np.nan))
    df["log_radius"] = np.log10(df["planet_radius_re"].replace(0, np.nan))
    df["depth_over_duration"] = df["depth_ppm"] / df["transit_duration_hours"]
    df["mission"] = df["mission"].fillna("UNKNOWN")
    df["mission_cat"] = df["mission"].astype(str)
    df = df.reset_index(drop=True)

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df["num_numeric"] = df[NUMERIC_FEATURES].notna().sum(axis=1)
    df = df[df["num_numeric"] >= min_numeric].reset_index(drop=True)
    df["label"] = df["disposition_std"].map(LABEL_MAP)
    return df


def save_canonical_catalog(df: pd.DataFrame, filename: str = "canonical_catalog.parquet") -> Path:
    path = DATA_DIR / filename
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":  # pragma: no cover
    frame = build_canonical_catalog()
    location = save_canonical_catalog(frame)
    print(f"Canonical catalog saved to {location}")
