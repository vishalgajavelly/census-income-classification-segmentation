from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_paths, setup_logging


def load_raw_with_columns(raw_path: Path, columns_path: Path) -> pd.DataFrame:
    """Load raw CSV (no header) and apply column names from a 1-column file."""
    raw = pd.read_csv(raw_path, header=None)
    cols = pd.read_csv(columns_path, header=None)

    column_names = (
        cols.iloc[:, 0]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )

    if raw.shape[1] != len(column_names):
        raise ValueError(
            f"Column mismatch: raw has {raw.shape[1]} cols but columns file has {len(column_names)} names."
        )

    raw.columns = column_names
    return raw


def clean_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Notebook-consistent cleaning:
    - do NOT drop duplicates (survey data), only log count
    - strip whitespace in object columns
    - "?" -> "Unknown"
    - "Not in universe*" -> "Not applicable"
    - numeric sentinels -> NaN:
        wage per hour == 9999 -> NaN
        capital gains == 99999 -> NaN
        dividends from stocks == 99999 -> NaN
    - fill NA in object columns with "Unknown"
    """
    num_dups = int(df.duplicated().sum())
    logger.info(
        "Duplicate rows (not dropped): %d (%.2f%%)", num_dups, 100.0 * num_dups / len(df)
    )

    df = df.copy()

    # ---- numeric sentinel -> NaN (notebook) ----
    if "wage per hour" in df.columns:
        df.loc[df["wage per hour"] == 9999, "wage per hour"] = np.nan
    if "capital gains" in df.columns:
        df.loc[df["capital gains"] == 99999, "capital gains"] = np.nan
    if "dividends from stocks" in df.columns:
        df.loc[df["dividends from stocks"] == 99999, "dividends from stocks"] = np.nan

    # ---- categorical normalization ----
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        # strip whitespace without turning NaN into "nan"
        df.loc[:, cat_cols] = df[cat_cols].astype("string").apply(lambda s: s.str.strip())

        # notebook mappings
        df = df.replace({"?": "Unknown"})
        not_app = {
            "Not in universe",
            "Not in universe or children",
            "Not in universe under 1 year old",
        }
        df = df.replace({v: "Not applicable" for v in not_app})

        # fill remaining missing categoricals like notebook
        df.loc[:, cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """target = 1 if label contains '50000+' else 0."""
    if "label" not in df.columns:
        raise KeyError("Expected column 'label' not found. Cannot create target.")

    out = df.copy()
    label = out["label"].astype(str).str.strip()
    out["target"] = label.str.contains(r"50000\+", regex=True).astype(int)
    return out


def save_processed(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data: %s", out_path)


def run(raw_path: Path, columns_path: Path, out_path: Path) -> None:
    logger = setup_logging()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    if not columns_path.exists():
        raise FileNotFoundError(f"Columns file not found: {columns_path}")

    logger.info("Loading raw data: %s", raw_path)
    logger.info("Loading columns : %s", columns_path)

    df = load_raw_with_columns(raw_path, columns_path)
    df = clean_dataframe(df, logger)
    df = add_target(df)

    logger.info("Target distribution: %s", df["target"].value_counts().to_dict())
    save_processed(df, out_path, logger)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare census dataset for modeling.")
    paths = get_paths()

    p.add_argument("--raw", type=Path, default=paths.raw / "census_bureau_data.csv")
    p.add_argument("--columns", type=Path, default=paths.raw / "census_bureau_columns.csv")
    p.add_argument("--out", type=Path, default=paths.processed / "census_clean.csv")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args.raw, args.columns, args.out)


if __name__ == "__main__":
    main()