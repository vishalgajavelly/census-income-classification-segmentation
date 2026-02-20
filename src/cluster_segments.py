# src/cluster_segments.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from .utils import ensure_dir, get_paths, save_json, seed_everything
from .features import build_features, infer_feature_types


def build_clustering_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
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
        sparse_threshold=0.3,
    )


def split_num_cat(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return num_cols, cat_cols


def weighted_rate(y: pd.Series, w: pd.Series) -> float:
    return float(np.average(y.astype(float), weights=w.astype(float)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train segmentation model (SVD + KMeans), notebook-aligned.")
    ap.add_argument("--data", type=str, default="", help="Processed CSV (default: data/processed/census_clean.csv)")
    ap.add_argument("--out_dir", type=str, default="", help="Output dir (default: artifacts/segments)")
    ap.add_argument("--k", type=int, default=6, help="Number of clusters (notebook used 6)")
    ap.add_argument("--svd_components", type=int, default=50, help="TruncatedSVD components (notebook used 50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--drop_for_clust",
        type=str,
        default="full or part time employment stat",
        help="Drop this column for clustering only (notebook did this).",
    )
    ap.add_argument(
        "--keep_cols",
        type=str,
        nargs="*",
        default=["target", "weight"],
        help="Columns to keep in cluster_assignments.csv if present.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)

    paths = get_paths()
    data_path = Path(args.data) if args.data else (paths.processed / "census_clean.csv")
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (paths.artifacts / "segments"))

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}. Run: python -m src.data_prep")

    df = pd.read_csv(data_path)
    df = build_features(df)

    # For clustering features: exclude label/target/weight always (like notebook)
    exclude = {"label", "target", "weight"}
    feature_cols, _, _ = infer_feature_types(df, exclude=exclude)

    # Drop one problematic categorical for clustering ONLY (notebook)
    if args.drop_for_clust in feature_cols:
        feature_cols = [c for c in feature_cols if c != args.drop_for_clust]

    num_cols, cat_cols = split_num_cat(df, feature_cols)

    print(f"[INFO] Clustering features: {len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")

    preprocess = build_clustering_preprocessor(num_cols, cat_cols)
    X_encoded = preprocess.fit_transform(df[feature_cols])

    n_features_enc = X_encoded.shape[1]
    n_svd = max(2, min(int(args.svd_components), n_features_enc - 1))
    svd = TruncatedSVD(n_components=n_svd, random_state=args.seed)
    X_clust_ready = svd.fit_transform(X_encoded)

    kmeans = KMeans(n_clusters=int(args.k), n_init=20, random_state=args.seed)
    labels = kmeans.fit_predict(X_clust_ready)

    # ---- save models ----
    joblib.dump(preprocess, out_dir / "preprocess_clust.joblib")
    joblib.dump(svd, out_dir / "svd.joblib")
    joblib.dump(kmeans, out_dir / "kmeans.joblib")

    # ---- assignments ----
    keep = [c for c in args.keep_cols if c in df.columns]
    out_assign = df[keep].copy() if keep else pd.DataFrame(index=df.index)
    out_assign["cluster"] = labels
    out_assign.to_csv(out_dir / "cluster_assignments.csv", index=False)

    # ---- summary: notebook-style (n, weight share, target rates) ----
    summary = (
        pd.DataFrame({"cluster": labels})
        .value_counts()
        .reset_index(name="n")
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    if "weight" in df.columns:
        wsum = df.groupby(labels)["weight"].sum()
        summary["weight_sum"] = summary["cluster"].map(wsum.to_dict()).astype(float)
        summary["weight_share"] = summary["weight_sum"] / float(df["weight"].sum())

    if "target" in df.columns:
        # unweighted
        tr_unw = df.groupby(labels)["target"].mean()
        summary["target_rate_unweighted"] = summary["cluster"].map(tr_unw.to_dict()).astype(float)

        # weighted
        if "weight" in df.columns:
            tr_w = df.groupby(labels).apply(lambda g: weighted_rate(g["target"], g["weight"]))
            summary["target_rate_weighted"] = summary["cluster"].map(tr_w.to_dict()).astype(float)

    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    meta: Dict[str, Any] = {
        "k": int(args.k),
        "svd_components": int(n_svd),
        "seed": int(args.seed),
        "dropped_for_clustering": args.drop_for_clust if args.drop_for_clust in df.columns else None,
        "n_rows": int(df.shape[0]),
        "n_features_raw": int(len(feature_cols)),
        "n_features_encoded": int(n_features_enc),
        "svd_explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "feature_cols": feature_cols,
    }
    save_json(meta, out_dir / "metadata.json")

    print("\n[OK] Segmentation artifacts written to:", out_dir.resolve())
    print("-", out_dir / "preprocess_clust.joblib")
    print("-", out_dir / "svd.joblib")
    print("-", out_dir / "kmeans.joblib")
    print("-", out_dir / "cluster_assignments.csv")
    print("-", out_dir / "cluster_summary.csv")
    print("-", out_dir / "metadata.json")


if __name__ == "__main__":
    main()