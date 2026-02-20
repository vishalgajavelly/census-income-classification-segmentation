from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from .utils import ensure_dir, get_paths, save_json, seed_everything
from .features import build_features, infer_feature_types


@dataclass
class TrainConfig:
    seed: int = 42
    threshold: float = 0.50
    test_size: float = 0.20
    xgb_params: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.seed,
            )


def make_preprocess(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )


def make_models(cfg: TrainConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=cfg.seed),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(**cfg.xgb_params)
    return models


def evaluate(
    y_true: np.ndarray,
    proba: np.ndarray,
    thr: float,
    sample_weight: np.ndarray | None = None,
) -> Dict[str, float]:
    pred = (proba >= thr).astype(int)
    return {
        "Threshold": float(thr),
        "ROC-AUC": float(roc_auc_score(y_true, proba, sample_weight=sample_weight)),
        "PR-AUC": float(average_precision_score(y_true, proba, sample_weight=sample_weight)),
        "F1": float(f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LR/RF/XGB with sample weights and save artifacts.")
    ap.add_argument("--data", type=str, default="", help="Processed CSV path (default: data/processed/census_clean.csv)")
    ap.add_argument("--out_dir", type=str, default="", help="Output dir (default: artifacts/models)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--test_size", type=float, default=0.20)
    args = ap.parse_args()

    paths = get_paths()
    data_path = Path(args.data) if args.data else (paths.processed / "census_clean.csv")
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (paths.artifacts / "models"))

    cfg = TrainConfig(seed=args.seed, threshold=args.threshold, test_size=args.test_size)
    seed_everything(cfg.seed)

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}. Run: python -m src.data_prep")

    df = pd.read_csv(data_path)

    # Feature engineering (safe if columns missing)
    df = build_features(df)

    if "target" not in df.columns:
        raise ValueError("Missing 'target' column. Did you run data_prep?")
    if "weight" not in df.columns:
        raise ValueError("Missing 'weight' column.")

    y = df["target"].astype(int).values
    w = df["weight"].astype(float).values

    # IMPORTANT: don't leak label/target/weight into features
    X_raw = df.drop(columns=[c for c in ["target", "label", "weight"] if c in df.columns])

    # Infer feature types on full X (keeps consistent column handling)
    feature_cols, numeric_features, categorical_features = infer_feature_types(X_raw)

    X = X_raw[feature_cols]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    print(f"[INFO] rows={len(df)} | features={len(feature_cols)} (num={len(numeric_features)}, cat={len(categorical_features)})")

    models = make_models(cfg)
    rows = {}
    metrics_rows = []

    for name, model in models.items():
        scale_numeric = (name == "Logistic Regression")
        preprocess = make_preprocess(numeric_features, categorical_features, scale_numeric=scale_numeric)

        pipe = Pipeline([("preprocess", preprocess), ("model", model)])

        print(f"[TRAIN] {name}")
        pipe.fit(X_train, y_train, model__sample_weight=w_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, proba, cfg.threshold, sample_weight=w_test)

        metrics_rows.append({"Model": name, **m})
        rows[name] = pipe
        print(f"  -> PR-AUC={m['PR-AUC']:.4f} ROC-AUC={m['ROC-AUC']:.4f} F1={m['F1']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["PR-AUC", "F1", "ROC-AUC"], ascending=False)
    best_name = str(metrics_df.iloc[0]["Model"])
    best_pipe = rows[best_name]
    
    best_model_path = out_dir / "best_model.joblib"
    joblib.dump(best_pipe, best_model_path)
    print(f"- {best_model_path}")
    
    # Save artifacts (keep minimal: metrics + meta; model save optional if you want)
    ensure_dir(out_dir)
    metrics_path = out_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    meta = {
        "best_model_name": best_name,
        "threshold": cfg.threshold,
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "features_total": int(len(feature_cols)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models_trained": list(models.keys()),
        "data_path": str(data_path),
    }
    save_json(meta, out_dir / "train_meta.json")

    print("\n[OK] Saved:")
    print(f"- {metrics_path}")
    print(f"- {out_dir / 'train_meta.json'}")
    print(f"[BEST] {best_name}")


if __name__ == "__main__":
    main()