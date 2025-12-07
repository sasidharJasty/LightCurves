"""1D CNN for phase-folded light-curve vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import ARTIFACTS_DIR, SEED, set_global_seeds


class PhaseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]).unsqueeze(0), torch.tensor(self.y[idx])


class CNN1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class CNNTrainingResult:
    best_f1: float
    model_path: str


def train_phase_cnn(
    X_phase: np.ndarray,
    y_phase: np.ndarray,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> CNNTrainingResult:
    set_global_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_phase,
        y_phase,
        test_size=0.2,
        stratify=y_phase,
        random_state=SEED,
    )
    train_loader = DataLoader(PhaseDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(PhaseDataset(X_va, y_va), batch_size=batch_size, shuffle=False, num_workers=2)

    model = CNN1D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    artifact_path = ARTIFACTS_DIR / "cnn_phase_best.pth"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                labels.append(yb.numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        f1 = f1_score(labels, preds, average="macro")
        print(f"Epoch {epoch:03d} train_loss={np.mean(losses):.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), artifact_path)
    return CNNTrainingResult(best_f1=best_f1, model_path=str(artifact_path))


if __name__ == "__main__":  # pragma: no cover
    from .phase_fold_pipeline import load_phase_data

    X_phase, y_phase = load_phase_data()
    if X_phase is None or y_phase is None:
        raise SystemExit("Phase-folded data not found. Generate it first.")
    train_phase_cnn(X_phase, y_phase)
