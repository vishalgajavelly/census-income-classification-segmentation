# src/features.py
from __future__ import annotations

from typing import Sequence, Tuple, List, Set

import numpy as np
import pandas as pd


def _safe_log1p_fill0(x: pd.Series) -> pd.Series:
    """Notebook-style: log1p(x.fillna(0))"""
    v = pd.to_numeric(x, errors="coerce").fillna(0)
    return np.log1p(v)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering aligned with EDA_modelling.ipynb:add_engineered_features.
    Adds engineered columns but does NOT drop originals.
    """
    out = df.copy()

    # normalize key string cols used in exact matches (notebook)
    str_cols = [
        "education",
        "marital stat",
        "citizenship",
        "class of worker",
        "full or part time employment stat",
        "country of birth father",
        "country of birth mother",
        "country of birth self",
    ]
    for c in str_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    # --- Education ordinal (NOTE: notebook mapping differs from your previous) ---
    if "education" in out.columns:
        edu_order = {
            "Children": 0,
            "Less than 1st grade": 1,
            "1st 2nd 3rd or 4th grade": 3,
            "5th or 6th grade": 6,
            "7th and 8th grade": 8,
            "9th grade": 9,
            "10th grade": 10,
            "11th grade": 11,
            "12th grade no diploma": 11,
            "High school graduate": 12,
            "Some college but no degree": 13,
            "Associates degree-occup /vocational": 14,
            "Associates degree-academic program": 14,
            "Bachelors degree(BA AB BS)": 16,
            "Masters degree(MA MS MEng MEd MSW MBA)": 18,
            "Prof school degree (MD DDS DVM LLB JD)": 19,
            "Doctorate degree(PhD EdD)": 20,
        }
        out["education_num"] = out["education"].map(edu_order)
        out["education_num_missing"] = out["education_num"].isna().astype("int8")

    # --- Age bins (notebook bins + labels) ---
    if "age" in out.columns:
        out["age_group"] = pd.cut(
            pd.to_numeric(out["age"], errors="coerce"),
            bins=[0, 15, 25, 35, 45, 55, 65, 100],
            labels=["child", "young", "early_career", "mid_career", "senior", "pre_retire", "retired"],
            include_lowest=True,
        ).astype("string")

    # --- Weeks worked bins (notebook bins + labels; includes 48 cut) ---
    if "weeks worked in year" in out.columns:
        out["weeks_worked_group"] = pd.cut(
            pd.to_numeric(out["weeks worked in year"], errors="coerce"),
            bins=[-1, 0, 13, 26, 39, 48, 52],
            labels=["none", "quarter", "half", "three_quarter", "most", "full_year"],
            include_lowest=True,
        ).astype("string")

    # --- Investment signals (notebook) ---
    # has_investment_income = (gains>0) OR (dividends>0)
    if {"capital gains", "dividends from stocks"}.issubset(out.columns):
        cg = pd.to_numeric(out["capital gains"], errors="coerce").fillna(0)
        dv = pd.to_numeric(out["dividends from stocks"], errors="coerce").fillna(0)
        out["has_investment_income"] = ((cg > 0) | (dv > 0)).astype("int8")

    # total_investment_income = gains + dividends - losses
    if {"capital gains", "dividends from stocks", "capital losses"}.issubset(out.columns):
        cg = pd.to_numeric(out["capital gains"], errors="coerce").fillna(0)
        dv = pd.to_numeric(out["dividends from stocks"], errors="coerce").fillna(0)
        cl = pd.to_numeric(out["capital losses"], errors="coerce").fillna(0)
        out["total_investment_income"] = cg + dv - cl

    # --- Log transforms (notebook) ---
    if "capital gains" in out.columns:
        out["log_capital_gains"] = _safe_log1p_fill0(out["capital gains"])
    if "dividends from stocks" in out.columns:
        out["log_dividends"] = _safe_log1p_fill0(out["dividends from stocks"])
    if "wage per hour" in out.columns:
        out["log_wage_per_hour"] = _safe_log1p_fill0(out["wage per hour"])

    # --- Earnings proxy (notebook) ---
    if "wage per hour" in out.columns and "weeks worked in year" in out.columns:
        wage = pd.to_numeric(out["wage per hour"], errors="coerce").fillna(0)
        weeks = pd.to_numeric(out["weeks worked in year"], errors="coerce").fillna(0)
        out["annual_earnings_proxy"] = wage * weeks

    # --- Flags (notebook exact categories) ---
    if "marital stat" in out.columns:
        out["is_married"] = out["marital stat"].isin(
            ["Married-civilian spouse present", "Married-A F spouse present"]
        ).astype("int8")

    if "citizenship" in out.columns:
        out["is_citizen"] = out["citizenship"].isin(
            [
                "Native- Born in the United States",
                "Native- Born in Puerto Rico or U S Outlying",
                "Native- Born abroad of American Parent(s)",
                "Foreign born- U S citizen by naturalization",
            ]
        ).astype("int8")

    if "class of worker" in out.columns:
        out["is_self_employed"] = out["class of worker"].isin(
            ["Self-employed-not incorporated", "Self-employed-incorporated"]
        ).astype("int8")

    if "full or part time employment stat" in out.columns:
        out["is_fulltime"] = (out["full or part time employment stat"] == "Full-time schedules").astype("int8")

    # --- Birthplace indicators (notebook) ---
    if {"country of birth father", "country of birth mother"}.issubset(out.columns):
        out["parents_both_us"] = (
            (out["country of birth father"] == "United-States")
            & (out["country of birth mother"] == "United-States")
        ).astype("int8")

    if "country of birth self" in out.columns:
        out["born_us"] = (out["country of birth self"] == "United-States").astype("int8")

    return out


def infer_feature_types(
    df: pd.DataFrame,
    exclude: Set[str] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (feature_cols, numeric_features, categorical_features).
    Excludes label/target/weight columns by default.
    """
    if exclude is None:
        exclude = {"label", "target", "weight"}

    candidate_cols = [c for c in df.columns if c not in exclude]

    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            categorical_features.append(c)

    feature_cols = numeric_features + categorical_features
    return feature_cols, numeric_features, categorical_features