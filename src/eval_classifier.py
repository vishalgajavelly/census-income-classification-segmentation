from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from .utils import ensure_dir, get_paths, load_json, save_json
from .features import build_features


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_proba_any(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))
    return np.asarray(model.predict(X), dtype=float)


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    thr: float,
    sample_weight: np.ndarray | None = None,
) -> Dict[str, float]:
    pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y_true, proba, sample_weight=sample_weight)),
        "pr_auc": float(average_precision_score(y_true, proba, sample_weight=sample_weight)),
        "precision": float(precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "recall": float(recall_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "f1": float(f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
    }


def weighted_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    tn = w[(y_true == 0) & (y_pred == 0)].sum()
    fp = w[(y_true == 0) & (y_pred == 1)].sum()
    fn = w[(y_true == 1) & (y_pred == 0)].sum()
    tp = w[(y_true == 1) & (y_pred == 1)].sum()
    return np.array([[tn, fp], [fn, tp]], dtype=float)


def save_confusion_report(out_path: Path, cm_unw: np.ndarray, cm_w: np.ndarray) -> None:
    lines = [
        "Confusion Matrix (unweighted counts)",
        "[[TN, FP],",
        " [FN, TP]]",
        str(cm_unw),
        "",
        "Confusion Matrix (weighted by sample weight)",
        "[[TN, FP],",
        " [FN, TP]]",
        np.array2string(cm_w, formatter={"float_kind": lambda x: f"{x:,.2f}"}),
        "",
    ]
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_cm_plot(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            text = f"{val:,.0f}" if float(val).is_integer() else f"{val:,.2f}"
            plt.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray, out_dir: Path, w: np.ndarray | None) -> Dict[str, Path]:
    ensure_dir(out_dir)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba, sample_weight=w)
    roc_auc = roc_auc_score(y_true, proba, sample_weight=w)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (weighted)")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, proba, sample_weight=w)
    pr_auc = average_precision_score(y_true, proba, sample_weight=w)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (weighted)")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower left")
    pr_path = out_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    return {"roc": roc_path, "pr": pr_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved classifier on the same split as training.")
    ap.add_argument("--data", type=str, default="", help="Processed CSV (default: data/processed/census_clean.csv)")
    ap.add_argument("--model_path", type=str, default="", help="Model path (default: artifacts/models/best_model.joblib)")
    ap.add_argument("--meta_path", type=str, default="", help="Train meta json (default: artifacts/models/train_meta.json)")
    ap.add_argument("--out_dir", type=str, default="", help="Output dir (default: artifacts/eval)")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    paths = get_paths()
    data_path = Path(args.data) if args.data else (paths.processed / "census_clean.csv")
    model_path = Path(args.model_path) if args.model_path else (paths.artifacts / "models" / "best_model.joblib")
    meta_path = Path(args.meta_path) if args.meta_path else (paths.artifacts / "models" / "train_meta.json")
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (paths.artifacts / "eval"))

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Train meta not found: {meta_path} (run train_classifier first)")

    meta = load_json(meta_path)
    seed = int(meta.get("seed", 42))
    test_size = float(meta.get("test_size", 0.20))

    df = pd.read_csv(data_path)
    df = build_features(df)

    if "target" not in df.columns or "weight" not in df.columns:
        raise ValueError("Expected 'target' and 'weight' in processed data.")

    y = df["target"].astype(int).values
    w = df["weight"].astype(float).values
    X = df.drop(columns=[c for c in ["target", "label", "weight"] if c in df.columns])

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=seed, stratify=y
    )
    # only evaluate on test split
    model = joblib.load(model_path)
    proba = predict_proba_any(model, X_test)

    metrics_w = compute_metrics(y_test, proba, args.threshold, sample_weight=w_test)
    metrics_unw = compute_metrics(y_test, proba, args.threshold, sample_weight=None)

    y_pred = (proba >= args.threshold).astype(int)
    cm_unw = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_w = weighted_confusion_matrix(y_test, y_pred, w_test)

    save_json({"weighted": metrics_w, "unweighted": metrics_unw}, out_dir / "metrics.json")
    pd.DataFrame(
        [{"setting": "weighted", **metrics_w}, {"setting": "unweighted", **metrics_unw}]
    ).to_csv(out_dir / "metrics.csv", index=False)

    save_confusion_report(out_dir / "confusion_matrix.txt", cm_unw, cm_w)
    save_cm_plot(cm_unw.astype(float), out_dir / "confusion_unweighted.png", "Confusion (unweighted)")
    save_cm_plot(cm_w, out_dir / "confusion_weighted.png", "Confusion (weighted)")

    curves = plot_roc_pr(y_test, proba, out_dir, w_test)

    if args.sweep:
        thr_grid = np.linspace(0.05, 0.95, 19)
        rows = [compute_metrics(y_test, proba, float(t), sample_weight=w_test) for t in thr_grid]
        pd.DataFrame(rows).to_csv(out_dir / "threshold_sweep.csv", index=False)

    print("[OK] Saved to:", out_dir)
    print("-", out_dir / "metrics.json")
    print("-", out_dir / "metrics.csv")
    print("-", out_dir / "confusion_matrix.txt")
    print("-", out_dir / "confusion_unweighted.png")
    print("-", out_dir / "confusion_weighted.png")
    print("-", curves["roc"])
    print("-", curves["pr"])


if __name__ == "__main__":
    main()