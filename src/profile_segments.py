# src/profile_segments.py
"""
Profile clusters into interpretable personas + generate report-ready tables/figures.

Notebook-aligned principles:
- Read the SAME processed dataset used for clustering: data/processed/census_clean.csv
- Recompute engineered features via build_features(df)
- Join cluster assignments by row order (consistent because clustering also runs on the same file without reordering)
- Use survey weights ("weight") for:
    - cluster weight_share
    - high-income propensity (weighted target rate)
    - weighted numeric means
    - weighted categorical top-K shares (IMPORTANT)

Reads:
- data/processed/census_clean.csv   (or --data override)
- artifacts/segments/cluster_assignments.csv (or --segments_dir override)

Writes (under --out_dir):
- segment_profile_table.csv
- segment_top_categories/*.csv
- persona_map.json
- figs/segment_numeric_heatmap.png
- figs/persona_bubble.png

Run:
python -m src.profile_segments \
  --segments_dir artifacts/segments \
  --out_dir artifacts/segments_profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, get_paths, save_json, seed_everything
from .features import build_features


# ----------------------------
# Weighted helpers
# ----------------------------
def wavg(series: pd.Series, weights: pd.Series) -> float:
    """Weighted average ignoring NaNs in series; ignores non-positive weights."""
    x = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = x.notna() & (w > 0)
    if mask.sum() == 0:
        return float(x.mean())
    return float(np.average(x[mask], weights=w[mask]))


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    z = scaler.fit_transform(df.values)
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def top_k_shares_unweighted(df: pd.DataFrame, group_col: str, cat_col: str, top_k: int = 3) -> pd.DataFrame:
    """Fallback if weights missing."""
    res: List[Dict[str, Any]] = []
    for cl, g in df.groupby(group_col, observed=True):
        vc = g[cat_col].fillna("Unknown").astype(str).value_counts(normalize=True)
        tops = vc.head(top_k)
        row: Dict[str, Any] = {"cluster": int(cl)}
        for i in range(top_k):
            row[f"top{i+1}"] = f"{tops.index[i]} ({float(tops.iloc[i]):.3f})" if i < len(tops) else ""
        res.append(row)
    return pd.DataFrame(res).sort_values("cluster").reset_index(drop=True)


def top_k_shares_weighted(
    df: pd.DataFrame,
    group_col: str,
    cat_col: str,
    weight_col: str,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    cluster | top1 | top2 | top3
    each top is 'category (weighted_share)'
    """
    res: List[Dict[str, Any]] = []
    for cl, g in df.groupby(group_col, observed=True):
        g2 = g[[cat_col, weight_col]].copy()
        g2[cat_col] = g2[cat_col].fillna("Unknown").astype(str)
        w = pd.to_numeric(g2[weight_col], errors="coerce").fillna(0.0)

        denom = float(w.sum())
        if denom <= 0:
            shares = g2[cat_col].value_counts(normalize=True)
        else:
            wsum = (
                g2.assign(_w=w)
                .groupby(cat_col, observed=True)["_w"]
                .sum()
                .sort_values(ascending=False)
            )
            shares = wsum / denom

        tops = shares.head(top_k)
        row = {"cluster": int(cl)}
        for i in range(top_k):
            row[f"top{i+1}"] = f"{tops.index[i]} ({float(tops.iloc[i]):.3f})" if i < len(tops) else ""
        res.append(row)

    return pd.DataFrame(res).sort_values("cluster").reset_index(drop=True)


# ----------------------------
# Persona naming (heuristic, report-friendly)
# ----------------------------
DEFAULT_PERSONA = {
    0: "Older Non-Workers",
    1: "Low-Income Workers",
    2: "Steady Workers",
    3: "Affluent Investors",
    4: "Dependents",
    5: "Prime Full-Time Workers",
}


def infer_personas(seg: pd.DataFrame) -> Dict[int, str]:
    """
    Simple heuristic persona naming using columns computed in segment_profile_table:
      - hi_rate_w
      - age
      - weeks worked in year
      - has_investment_income
      - is_fulltime
    """
    personas: Dict[int, str] = {}
    overall_hi = float(seg["hi_rate_w"].mean()) if "hi_rate_w" in seg.columns else 0.0

    # rank by income propensity
    rank = seg.sort_values("hi_rate_w", ascending=False).copy()

    for _, r in rank.iterrows():
        c = int(r["cluster"])
        hi = float(r.get("hi_rate_w", np.nan))
        age = float(r.get("age", np.nan))
        weeks = float(r.get("weeks worked in year", np.nan))
        inv = float(r.get("has_investment_income", np.nan))
        ft = float(r.get("is_fulltime", np.nan))

        if (not np.isnan(weeks) and weeks <= 1) and (not np.isnan(hi) and hi < 0.02):
            personas[c] = "Dependents"
            continue

        if (not np.isnan(hi) and hi >= max(0.20, overall_hi * 2.0)) and (not np.isnan(inv) and inv >= 0.30):
            personas[c] = "Affluent Investors"
            continue

        if (not np.isnan(age) and age >= 55) and (not np.isnan(weeks) and weeks < 10) and (not np.isnan(hi) and hi < 0.05):
            personas[c] = "Older Non-Workers"
            continue

        if not np.isnan(hi) and hi >= overall_hi * 1.2:
            personas[c] = "Prime Full-Time Workers" if (not np.isnan(ft) and ft >= 0.6) else "Steady Workers"
        elif not np.isnan(hi) and hi >= overall_hi * 0.8:
            personas[c] = "Steady Workers"
        else:
            personas[c] = "Low-Income Workers"

    for c in seg["cluster"].astype(int).unique():
        personas.setdefault(int(c), DEFAULT_PERSONA.get(int(c), f"Cluster {int(c)}"))

    return personas


# ----------------------------
# Plots (matplotlib only)
# ----------------------------
def plot_numeric_heatmap(seg_num_z: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.6 * len(seg_num_z))))
    im = ax.imshow(seg_num_z.values, aspect="auto")

    ax.set_yticks(range(len(seg_num_z.index)))
    ax.set_yticklabels([str(i) for i in seg_num_z.index])

    ax.set_xticks(range(len(seg_num_z.columns)))
    ax.set_xticklabels(seg_num_z.columns, rotation=45, ha="right")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="z-score (weighted mean)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_persona_bubble(seg: pd.DataFrame, out_png: Path, overall_hi: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    if "weight_share" in seg.columns and seg["weight_share"].notna().any():
        sizes = 9000 * seg["weight_share"].clip(lower=0.001)
        x = seg["n"]
        xlab = "Segment size (n)"
    else:
        sizes = 9000 * (seg["n"] / seg["n"].sum()).clip(lower=0.001)
        x = seg["n"]
        xlab = "Segment size (n)"

    ax.scatter(x, seg["hi_rate_w"], s=sizes, alpha=0.6)

    for _, r in seg.iterrows():
        ax.text(
            float(r["n"]),
            float(r["hi_rate_w"]),
            f"C{int(r['cluster'])}: {r['persona']}",
            fontsize=8,
            ha="left",
            va="center",
        )

    ax.axhline(overall_hi, linestyle="--", color="black", alpha=0.7, label="Overall (weighted)")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Weighted P(Income ≥ $50K)")
    ax.set_title("Segment Personas: Size vs High-Income Propensity")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Profile clusters into personas + visuals (notebook-aligned)")
    parser.add_argument("--data", type=str, default="", help="Processed CSV (default: data/processed/census_clean.csv)")
    parser.add_argument("--segments_dir", type=str, default="", help="Segments dir (default: artifacts/segments)")
    parser.add_argument("--out_dir", type=str, default="", help="Output dir (default: artifacts/segments_profile)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_col", type=str, default="target")
    parser.add_argument("--weight_col", type=str, default="weight")
    parser.add_argument("--cluster_col", type=str, default="cluster")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--persona_json", type=str, default=None)

    parser.add_argument(
        "--cat_profile_cols",
        type=str,
        nargs="*",
        default=[
            "education",
            "class of worker",
            "marital stat",
            "sex",
            "race",
            "major occupation code",
            "tax filer stat",
        ],
    )
    parser.add_argument(
        "--num_profile_cols",
        type=str,
        nargs="*",
        default=[
            "age",
            "education_num",
            "weeks worked in year",
            "log_wage_per_hour",
            "has_investment_income",
            "is_fulltime",
        ],
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    paths = get_paths()
    data_path = Path(args.data) if args.data else (paths.processed / "census_clean.csv")
    segments_dir = Path(args.segments_dir) if args.segments_dir else (paths.artifacts / "segments")
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (paths.artifacts / "segments_profile"))

    figs_dir = ensure_dir(out_dir / "figs")
    topcats_dir = ensure_dir(out_dir / "segment_top_categories")

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}. Run: python -m src.data_prep")

    assign_fp = segments_dir / "cluster_assignments.csv"
    if not assign_fp.exists():
        raise FileNotFoundError(f"Missing {assign_fp}. Run: python -m src.cluster_segments")

    # load processed data (same dataset as clustering)
    df = pd.read_csv(data_path)
    df = build_features(df)

    # load cluster assignments (same row order)
    assignments = pd.read_csv(assign_fp)
    if args.cluster_col not in assignments.columns:
        raise ValueError(f"{assign_fp} must contain '{args.cluster_col}'.")

    if len(assignments) != len(df):
        raise ValueError(
            f"Row mismatch: assignments has {len(assignments)}, data has {len(df)}. "
            "Ensure cluster_segments was run on the same census_clean.csv without reordering."
        )

    df_seg = df.copy()
    df_seg[args.cluster_col] = assignments[args.cluster_col].astype(int).values

    has_weight = args.weight_col in df_seg.columns
    wcol = args.weight_col

    # Build segment summary table
    rows: List[Dict[str, Any]] = []
    for cl, g in df_seg.groupby(args.cluster_col, observed=True):
        cl = int(cl)
        row: Dict[str, Any] = {"cluster": cl, "n": int(len(g))}

        if has_weight:
            w = pd.to_numeric(g[wcol], errors="coerce").fillna(0.0)
            row["weight_sum"] = float(w.sum())
            denom = float(pd.to_numeric(df_seg[wcol], errors="coerce").fillna(0.0).sum())
            row["weight_share"] = row["weight_sum"] / denom if denom > 0 else np.nan

            if args.target_col in g.columns:
                row["hi_rate_w"] = wavg(g[args.target_col], g[wcol])
            else:
                row["hi_rate_w"] = np.nan
        else:
            row["weight_sum"] = np.nan
            row["weight_share"] = np.nan
            row["hi_rate_w"] = float(pd.to_numeric(g.get(args.target_col, pd.Series(dtype=float)), errors="coerce").mean()) if args.target_col in g.columns else np.nan

        # numeric profiles
        for c in args.num_profile_cols:
            if c in g.columns:
                row[c] = wavg(g[c], g[wcol]) if has_weight else float(pd.to_numeric(g[c], errors="coerce").mean())

        rows.append(row)

    seg = pd.DataFrame(rows)

    # Personas
    if args.persona_json:
        persona_map = json.loads(Path(args.persona_json).read_text(encoding="utf-8"))
        persona_map = {int(k): str(v) for k, v in persona_map.items()}
    else:
        persona_map = infer_personas(seg)

    seg["persona"] = seg["cluster"].map(persona_map)
    seg = seg.sort_values("hi_rate_w", ascending=False).reset_index(drop=True)

    # Save persona + profile table
    save_json(persona_map, out_dir / "persona_map.json")
    seg.to_csv(out_dir / "segment_profile_table.csv", index=False)

    # Top categories per cluster (WEIGHTED if weight exists)
    for col in args.cat_profile_cols:
        if col in df_seg.columns:
            if has_weight:
                top_df = top_k_shares_weighted(df_seg, args.cluster_col, col, wcol, top_k=args.top_k)
            else:
                top_df = top_k_shares_unweighted(df_seg, args.cluster_col, col, top_k=args.top_k)
            safe_name = col.replace(" ", "_")
            top_df.to_csv(topcats_dir / f"top_{safe_name}.csv", index=False)

    # Numeric heatmap (z-scored weighted means)
    heat_cols = [c for c in args.num_profile_cols if c in seg.columns and pd.api.types.is_numeric_dtype(seg[c])]
    if heat_cols:
        heat_base = seg.set_index("cluster")[heat_cols].copy()
        heat_z = zscore_df(heat_base)
        plot_numeric_heatmap(
            heat_z,
            out_png=figs_dir / "segment_numeric_heatmap.png",
            title="Segment numeric profile (z-scored weighted means)",
        )

    # Persona bubble
    if args.target_col in df_seg.columns:
        overall_hi = wavg(df_seg[args.target_col], df_seg[wcol]) if has_weight else float(pd.to_numeric(df_seg[args.target_col], errors="coerce").mean())
        plot_persona_bubble(seg, out_png=figs_dir / "persona_bubble.png", overall_hi=float(overall_hi))

    print("✅ Segment profiling complete. Outputs written to:", out_dir.resolve())
    print("   - segment_profile_table.csv")
    print("   - persona_map.json")
    print("   - segment_top_categories/*.csv")
    print("   - figs/segment_numeric_heatmap.png (if numeric cols available)")
    print("   - figs/persona_bubble.png")


if __name__ == "__main__":
    main()