"""
Fraud Transaction Detection - SMOTE Comparison
Compares RandomForest performance with and without SMOTE for imbalanced datasets.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker/headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Paths relative to project root (works from src/ or project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Fraud_Transaction_Detection.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "static"

# Columns to drop (PII / non-predictive)
DROP_COLUMNS = ["Customer Name", "Customer Email", "Customer Phone", "Transaction ID"]


# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load and validate fraud detection dataset. Uses demo data if file not found."""
    if not path.exists():
        # Generate demo data for showcase/CI when real data is unavailable
        path.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.generate_demo_data import generate_demo_data
        df = generate_demo_data()
        df.to_excel(path, index=False)
        print(f"Using demo data (real file not found): {path}")
        return df
    df = pd.read_excel(path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def encode_categorical(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Encode categorical columns. Fits encoders on training data only to prevent
    data leakage from test set. Unseen categories in test get encoded as -1.
    """
    encoders = {}
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    return X_train, X_test, encoders


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop unnecessary columns and prepare X, y."""
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    if "Is_Fraud" not in df.columns:
        raise ValueError("Target column 'Is_Fraud' not found in dataset")
    X = df.drop(columns=["Is_Fraud"])
    y = df["Is_Fraud"]
    return X, y


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_class_distribution(y: pd.Series, filepath: Path, title: str, palette: str = "pastel") -> None:
    """Plot and save class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette=palette)
    plt.title(title)
    plt.xlabel("Class (0 = Not Fraud, 1 = Fraud)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, filepath: Path, title: str, cmap: str = "Blues") -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, auc_score: float, filepath: Path, title: str, color: str = "blue"
) -> None:
    """Plot and save ROC curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color=color)
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# -----------------------------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------------------------
def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """Compute metrics and ROC curve data for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": auc(fpr, tpr),
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def print_results(label: str, metrics: dict, use_smote: bool) -> None:
    """Print formatted model results."""
    emoji = "✅" if use_smote else "❌"
    print("=" * 50)
    print(f"{emoji} {label}")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:  {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    """Run fraud detection pipeline: load data, train models, compare SMOTE vs no-SMOTE."""
    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data (use demo data if real dataset not found - for showcase/CI)
    if DATA_PATH.exists():
        df = load_data(DATA_PATH)
    else:
        print("Data file not found. Generating demo data for showcase...", file=sys.stderr)
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.generate_demo_data import generate_demo_data
        df = generate_demo_data(n_samples=2000, output_path=DATA_PATH)

    X, y = preprocess_data(df)

    # Split before encoding to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Encode categoricals (fit on train only)
    X_train, X_test, _ = encode_categorical(X_train, X_test)

    # Ensure numeric types for any remaining object columns
    for col in X_train.select_dtypes(include=["object"]).columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(-1)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(-1)

    # Original class distribution
    plot_class_distribution(
        y,
        OUTPUT_DIR / "original_class_dist.png",
        "Original Class Distribution (0 = Not Fraud, 1 = Fraud)",
    )

    # -------------------------------------------------------------------------
    # Model WITHOUT SMOTE
    # -------------------------------------------------------------------------
    clf_original = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE
    )
    clf_original.fit(X_train, y_train)
    metrics_original = evaluate_model(clf_original, X_test, y_test)

    plot_confusion_matrix(
        metrics_original["confusion_matrix"],
        OUTPUT_DIR / "confusion_matrix_no_smote.png",
        "Confusion Matrix (Without SMOTE)",
        cmap="Purples",
    )
    plot_roc_curve(
        metrics_original["fpr"],
        metrics_original["tpr"],
        metrics_original["roc_auc"],
        OUTPUT_DIR / "roc_curve_no_smote.png",
        "ROC Curve (Without SMOTE)",
        color="purple",
    )
    print_results("Model WITHOUT SMOTE", metrics_original, use_smote=False)

    # -------------------------------------------------------------------------
    # Model WITH SMOTE
    # -------------------------------------------------------------------------
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    plot_class_distribution(
        y_train_resampled,
        OUTPUT_DIR / "smote_class_dist.png",
        "SMOTE Class Distribution (0 = Not Fraud, 1 = Fraud)",
        palette="muted",
    )

    clf_smote = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE
    )
    clf_smote.fit(X_train_resampled, y_train_resampled)
    metrics_smote = evaluate_model(clf_smote, X_test, y_test)

    plot_confusion_matrix(
        metrics_smote["confusion_matrix"],
        OUTPUT_DIR / "confusion_matrix.png",
        "Confusion Matrix (With SMOTE)",
    )
    plot_roc_curve(
        metrics_smote["fpr"],
        metrics_smote["tpr"],
        metrics_smote["roc_auc"],
        OUTPUT_DIR / "roc_curve.png",
        "ROC Curve (With SMOTE)",
    )
    print_results("Model WITH SMOTE", metrics_smote, use_smote=True)

    # -------------------------------------------------------------------------
    # Comparison CSV
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision (macro)",
                "Recall (macro)",
                "F1-Score (macro)",
                "AUC-ROC",
            ],
            "Without SMOTE": [
                metrics_original["accuracy"],
                metrics_original["precision"],
                metrics_original["recall"],
                metrics_original["f1"],
                metrics_original["roc_auc"],
            ],
            "With SMOTE": [
                metrics_smote["accuracy"],
                metrics_smote["precision"],
                metrics_smote["recall"],
                metrics_smote["f1"],
                metrics_smote["roc_auc"],
            ],
        }
    )
    comparison_path = OUTPUT_DIR / "smote_vs_no_smote_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print(f"\n📊 Comparison saved to: {comparison_path}")
    print(f"📁 Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
