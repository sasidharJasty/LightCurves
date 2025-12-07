"""Project-level configuration and reproducibility helpers."""
from __future__ import annotations

from pathlib import Path
import random
import numpy as np

try:  # Optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "exosphereai"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
for path in (ROOT, DATA_DIR, ARTIFACTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

SEED = 42


def set_global_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and Torch RNGs for repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "PROJECT_ROOT",
    "ROOT",
    "DATA_DIR",
    "ARTIFACTS_DIR",
    "SEED",
    "set_global_seeds",
]
