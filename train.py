from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src_pipeline import (
    BASE_FEATURES,
    FAILURE_MODE_COLUMNS,
    TARGET,
    add_engineered_features,
    build_preprocessor,
    choose_f2_threshold,
    classification_metrics,
    save_bundle,
    save_json,
)

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020_offline_replica.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
for path in (RESULTS, FIGURES, MODELS):
    path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    X = df[BASE_FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_models(y_train: pd.Series):
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    return {
        "Dummy baseline": Pipeline(
            [
                ("preprocess", build_preprocessor(use_engineered=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("preprocess", build_preprocessor(use_engineered=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", build_preprocessor(use_engineered=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", build_preprocessor(use_engineered=True)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        min_child_weight=2,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        reg_lambda=2.0,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def plot_class_balance(df: pd.DataFrame):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(["Normal", "Failure"], counts.values)
    ax.set_title("Class balance")
    ax.set_ylabel("Observations")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 80, f"{value:,}\n({value/len(df):.1%})", ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_balance.png", dpi=180, transparent=True)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame):
    # Two interpretable derived features that relate to published AI4I failure mechanisms.
    d = add_engineered_features(df)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for label, group in d.groupby(TARGET):
        ax.hist(
            group["Mechanical power [W]"],
            bins=45,
            alpha=0.55,
            density=True,
            label="Failure" if label else "Normal",
        )
    ax.set_title("Mechanical power distribution")
    ax.set_xlabel("Mechanical power [W]")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "power_distribution.png", dpi=180, transparent=True)
    plt.close(fig)


def main():
    df = load_data()
    plot_class_balance(df)
    plot_feature_distributions(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_eng = add_engineered_features(X_train)
    X_val_eng = add_engineered_features(X_val)
    X_test_eng = add_engineered_features(X_test)

    split_summary = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "rows": [len(X_train), len(X_val), len(X_test)],
            "failures": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
            "failure_rate": [y_train.mean(), y_val.mean(), y_test.mean()],
        }
    )
    split_summary.to_csv(RESULTS / "split_summary.csv", index=False)

    models = make_models(y_train)
    results = []
    fitted = {}
    thresholds = {}
    test_scores = {}

    for name, model in models.items():
        engineered = name != "Dummy baseline"
        train_x = X_train_eng if engineered else X_train
        val_x = X_val_eng if engineered else X_val
        test_x = X_test_eng if engineered else X_test

        model.fit(train_x, y_train)
        val_scores = model.predict_proba(val_x)[:, 1]
        threshold = 0.5 if name == "Dummy baseline" else choose_f2_threshold(y_val, val_scores)
        scores = model.predict_proba(test_x)[:, 1]
        metrics = classification_metrics(name, y_test, scores, threshold)
        results.append(metrics)
        fitted[name] = model
        thresholds[name] = threshold
        test_scores[name] = scores

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(RESULTS / "model_metrics.csv", index=False)

    # Feature-engineering ablation for XGBoost (same split; no hidden failure-mode labels used).
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    ablation_rows = []
    for label, engineered in [("Raw 6 features", False), ("+ 3 engineered features", True)]:
        model = Pipeline(
            [
                ("preprocess", build_preprocessor(use_engineered=engineered)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_weight=2,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        tr = X_train_eng if engineered else X_train
        va = X_val_eng if engineered else X_val
        te = X_test_eng if engineered else X_test
        model.fit(tr, y_train)
        val_scores = model.predict_proba(va)[:, 1]
        threshold = choose_f2_threshold(y_val, val_scores)
        test_score = model.predict_proba(te)[:, 1]
        row = classification_metrics(label, y_test, test_score, threshold)
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(RESULTS / "feature_ablation.csv", index=False)

    # Model metric chart.
    chart = metrics_df[metrics_df["model"] != "Dummy baseline"].copy()
    x = np.arange(len(chart))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width, chart["precision"], width, label="Precision")
    ax.bar(x, chart["recall"], width, label="Recall")
    ax.bar(x + width, chart["pr_auc"], width, label="PR-AUC")
    ax.set_xticks(x, chart["model"], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics")
    ax.legend(ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=180, transparent=True)
    plt.close(fig)

    # PR curves; prevalence shown as a horizontal reference line.
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    prevalence = y_test.mean()
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        precision, recall, _ = precision_recall_curve(y_test, test_scores[name])
        ap = average_precision_score(y_test, test_scores[name])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.axhline(prevalence, linestyle="--", linewidth=1.2, label=f"Prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve on the locked test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curve.png", dpi=180, transparent=True)
    plt.close(fig)

    # Final model: XGBoost because it has the strongest ranking metric (PR-AUC).
    final_name = "XGBoost"
    final_model = fitted[final_name]
    final_threshold = thresholds[final_name]
    final_scores = test_scores[final_name]
    final_pred = final_scores >= final_threshold

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        final_pred,
        display_labels=["Normal", "Failure"],
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"XGBoost confusion matrix\nthreshold={final_threshold:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_xgboost.png", dpi=180, transparent=True)
    plt.close(fig)

    # Explainability with SHAP on transformed test data.
    preprocessor = final_model.named_steps["preprocess"]
    estimator = final_model.named_steps["model"]
    transformed_test = preprocessor.transform(X_test_eng)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_importance.to_csv(RESULTS / "shap_importance.csv", index=False)

    top = shap_importance.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(top["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False), top["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature importance — XGBoost")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_importance.png", dpi=180, transparent=True)
    plt.close(fig)

    # Error analysis by hidden failure mode. These labels are never used as predictive inputs.
    test_detail = df.loc[X_test.index].copy()
    test_detail["risk_score"] = final_scores
    test_detail["prediction"] = final_pred.astype(int)
    rows = []
    for mode in FAILURE_MODE_COLUMNS:
        mask = test_detail[mode] == 1
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "failure_mode": mode,
                "test_cases": int(mask.sum()),
                "detected": int(test_detail.loc[mask, "prediction"].sum()),
                "recall": float(test_detail.loc[mask, "prediction"].mean()),
                "mean_risk_score": float(test_detail.loc[mask, "risk_score"].mean()),
            }
        )
    mode_df = pd.DataFrame(rows)
    mode_df.to_csv(RESULTS / "failure_mode_error_analysis.csv", index=False)

    false_negatives = test_detail[(test_detail[TARGET] == 1) & (test_detail["prediction"] == 0)]
    false_positives = test_detail[(test_detail[TARGET] == 0) & (test_detail["prediction"] == 1)]
    false_negatives.to_csv(RESULTS / "false_negatives.csv", index=False)
    false_positives.to_csv(RESULTS / "false_positives.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(mode_df["failure_mode"], mode_df["recall"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Which failure modes are detected?")
    for i, row in mode_df.iterrows():
        ax.text(i, row["recall"] + 0.03, f"{row['detected']}/{row['test_cases']}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "failure_mode_recall.png", dpi=180, transparent=True)
    plt.close(fig)

    # Reproducible inference bundle. Output is called a risk score because class weighting means it is not calibrated.
    metadata = {
        "model": final_name,
        "features": BASE_FEATURES,
        "engineered_features": [
            "Temperature gap [K]",
            "Mechanical power [W]",
            "Wear × torque",
        ],
        "threshold": final_threshold,
        "seed": SEED,
        "dataset": "ai4i2020_offline_replica.csv",
        "warning": "Risk score is not a calibrated probability and must not trigger autonomous maintenance actions.",
    }
    save_bundle(MODELS / "xgboost_failure_risk.joblib", final_model, final_threshold, metadata)

    summary = {
        "rows": len(df),
        "failures": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "split": {r["split"]: int(r["rows"]) for _, r in split_summary.iterrows()},
        "final_model": final_name,
        "final_threshold": float(final_threshold),
        "final_test_metrics": metrics_df.loc[metrics_df["model"] == final_name].iloc[0].to_dict(),
        "note": "Results are produced on the bundled offline AI4I-style replica, not the original UCI CSV.",
    }
    # Convert numpy scalar values from dataframe dict.
    summary["final_test_metrics"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in summary["final_test_metrics"].items()}
    save_json(RESULTS / "run_summary.json", summary)

    print(metrics_df.to_string(index=False))
    print("\nFinal threshold:", final_threshold)
    print("\nFailure-mode error analysis:\n", mode_df.to_string(index=False))
    print("\nTop SHAP features:\n", shap_importance.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
