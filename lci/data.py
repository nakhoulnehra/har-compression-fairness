"""Load the raw Phishing Websites CSVs and apply the authors' preprocessing + split (spec 2.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .paths import PHISHING_DIR

SPLIT_SEED = 42  # always 42, independent of the training seed


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_phishing_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    f, t = PHISHING_DIR / "features.csv", PHISHING_DIR / "targets.csv"
    if not (f.exists() and t.exists()):
        # No synthetic fallback, ever.
        raise FileNotFoundError(f"Phishing data not found in {PHISHING_DIR}; run `python data/download.py`")
    features, targets = pd.read_csv(f), pd.read_csv(t)
    assert features.shape == (11055, 30), features.shape
    assert targets.shape[0] == 11055, targets.shape
    assert targets.iloc[:, 0].nunique() == 2
    return features, targets


def preprocess(features: pd.DataFrame, targets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = features.copy()
    # 1. label-encode object/category columns (none in Phishing, kept for fidelity)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    # 2. numeric only
    df = df.select_dtypes(include=[np.number])
    # 3. drop constant columns
    df = df.loc[:, df.std() > 0]
    # 4. +-inf -> NaN -> 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # 5. scaler fit on the WHOLE dataset (before the split, as the authors did)
    X = StandardScaler().fit_transform(df.values).astype(np.float32)
    # 6. labels: -1 -> 0 (phishing), 1 -> 1 (legitimate)
    y = LabelEncoder().fit_transform(targets.values.ravel()).astype(np.int64)
    return X, y, list(df.columns)


def split_70_15_15(X: np.ndarray, y: np.ndarray, seed: int = SPLIT_SEED):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def load_phishing_split() -> Split:
    X, y, names = preprocess(*load_phishing_raw())
    return Split(*split_70_15_15(X, y), feature_names=names)
