from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED_FEATURES = [
    "Temperature gap [K]",
    "Mechanical power [W]",
    "Wear × torque",
]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"


def generate_ai4i_style_replica(
    seed: int = 42,
    n: int = 10_000,
    rho: float = -0.95,
    process_seed: int = 263,
    tool_failure_probability: float = 0.40,
    random_failure_probability: float = 0.0019,
) -> pd.DataFrame:
    """Generate an offline AI4I-style dataset using the public UCI generation rules.

    This is intentionally labelled a *replica*, not the original UCI CSV.
    It is bundled so the project can be executed without internet access.
    """
    rng = np.random.default_rng(seed)
    machine_type = rng.choice(["L", "M", "H"], n, p=[0.5, 0.3, 0.2])

    air_rng = np.random.default_rng(seed + 1)
    air_rw = np.cumsum(air_rng.normal(size=n))
    air = 300 + 2 * (air_rw - air_rw.mean()) / air_rw.std()

    process_rng = np.random.default_rng(process_seed)
    process_rw = np.cumsum(process_rng.normal(size=n))
    process_component = (process_rw - process_rw.mean()) / process_rw.std()
    process = air + 10 + process_component

    # Strong inverse RPM/torque relationship approximates a near-constant power regime.
    z_torque = rng.normal(size=n)
    z_rpm = rho * z_torque + np.sqrt(1 - rho**2) * rng.normal(size=n)
    torque = np.clip(40 + 10 * z_torque, 3, 80)
    rpm = np.clip(np.round(1538 + 179 * z_rpm), 1100, 2900).astype(int)

    wear_increment = {"L": 2, "M": 3, "H": 5}
    tool_wear = np.zeros(n, dtype=int)
    twf = np.zeros(n, dtype=int)
    current_wear = 0
    replacement_threshold = int(rng.integers(200, 241))
    for i, t in enumerate(machine_type):
        tool_wear[i] = current_wear
        current_wear += wear_increment[t]
        if current_wear >= replacement_threshold:
            twf[i] = int(rng.random() < tool_failure_probability)
            current_wear = 0
            replacement_threshold = int(rng.integers(200, 241))

    temperature_gap = process - air
    hdf = (temperature_gap < 8.6) & (rpm < 1380)

    mechanical_power = torque * (rpm * 2 * np.pi / 60)
    pwf = (mechanical_power < 3500) | (mechanical_power > 9000)

    overstrain_limit = np.select(
        [machine_type == "L", machine_type == "M", machine_type == "H"],
        [11_000, 12_000, 13_000],
    )
    osf = tool_wear * torque > overstrain_limit

    rnf = rng.random(n) < random_failure_probability
    machine_failure = twf.astype(bool) | hdf | pwf | osf | rnf

    product_id = np.array([f"{t}{10_000 + i:05d}" for i, t in enumerate(machine_type)])

    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": product_id,
            "Type": machine_type,
            "Air temperature [K]": np.round(air, 1),
            "Process temperature [K]": np.round(process, 1),
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": np.round(torque, 1),
            "Tool wear [min]": tool_wear,
            TARGET: machine_failure.astype(int),
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Temperature gap [K]"] = (
        frame["Process temperature [K]"] - frame["Air temperature [K]"]
    )
    frame["Mechanical power [W]"] = (
        frame["Torque [Nm]"]
        * frame["Rotational speed [rpm]"]
        * 2
        * np.pi
        / 60
    )
    frame["Wear × torque"] = frame["Tool wear [min]"] * frame["Torque [Nm]"]
    return frame


def build_preprocessor(use_engineered: bool = True) -> ColumnTransformer:
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    if use_engineered:
        numeric_features += ENGINEERED_FEATURES

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, ["Type"]),
        ]
    )


def choose_f2_threshold(y_true: Iterable[int], scores: np.ndarray) -> float:
    thresholds = np.linspace(0.02, 0.90, 177)
    return float(
        max(
            thresholds,
            key=lambda threshold: fbeta_score(
                y_true, scores >= threshold, beta=2, zero_division=0
            ),
        )
    )


def classification_metrics(
    name: str,
    y_true: Iterable[int],
    scores: np.ndarray,
    threshold: float,
) -> dict:
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_bundle(path: str | Path, pipeline: Pipeline, threshold: float, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": float(threshold),
            "metadata": metadata,
        },
        path,
    )


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
