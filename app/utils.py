"""Shared loading, feature engineering and styling for the Streamlit app.

The feature engineering in `build_features` must stay identical to Section 4b
of the notebook, because the saved pipeline expects exactly those columns.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "homes_2025_clean.csv"
MODEL_FILE = ROOT / "models" / "final_model.joblib"
RESULTS_DIR = ROOT / "models" / "results"

SALE_YEAR = 2025
FINAL_MODEL_NAME = "Gradient Boosting"

# Fixed colour per borough, in a fixed order, used on every page
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
BOROUGH_COLORS = dict(zip(BOROUGHS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
HOME_TYPES = ["1-family", "2-family", "3-family"]
HIGHLIGHT = "#2a78d6"   # the final model / the main series
MUTED = "#a3a29c"       # everything it is compared against

# NYC Department of Finance building class codes found in the data
BUILDING_CLASSES = {
    "A0": "Cape Cod style", "A1": "Two-storey detached", "A2": "One-storey",
    "A3": "Large suburban residence", "A4": "City residence", "A5": "Attached or semi-detached",
    "A6": "Summer cottage", "A7": "Mansion", "A8": "Bungalow colony", "A9": "Other one-family",
    "B1": "Two-family, brick", "B2": "Two-family, frame", "B3": "Two-family, converted from one-family",
    "B9": "Other two-family", "C0": "Three-family",
    "S0": "One-family with two stores or offices", "S1": "One-family with one store or office",
    "S2": "Two-family with one store or office",
}

FEATURE_LABELS = {
    "neighborhood": "Neighbourhood", "log_gross_sqft": "Building size", "log_land_sqft": "Land size",
    "borough_name": "Borough", "building_age": "Building age", "home_type": "Home type",
    "building_class_at_time_of_sale": "Building class", "floor_area_ratio": "Floor area ratio",
    "sale_quarter": "Sale quarter", "has_commercial": "Has a shop/office",
}


@st.cache_data
def load_homes() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, index_col="row_id", parse_dates=["sale_date"])
    df["sale_month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()
    df["neighborhood_title"] = df["neighborhood"].str.title()
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_results(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{name}.csv", index_col=0)


def build_features(borough, neighborhood, home_type, building_class,
                   gross_sqft, land_sqft, year_built, sale_quarter) -> pd.DataFrame:
    """One-row model input, engineered exactly as in Section 4b of the notebook."""
    gross = np.nan if gross_sqft is None else float(gross_sqft)
    land = np.nan if land_sqft is None else float(land_sqft)
    return pd.DataFrame([{
        "log_gross_sqft": np.log(gross),
        "log_land_sqft": np.log(land),
        "floor_area_ratio": gross / land,
        "building_age": max(SALE_YEAR - int(year_built), 0),
        # in the data a home has a commercial unit exactly when its class starts with S
        "has_commercial": float(building_class.startswith("S")),
        "home_type": home_type,
        "borough_name": borough,
        "building_class_at_time_of_sale": building_class,
        "sale_quarter": int(sale_quarter),
        "neighborhood": neighborhood,
    }])


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x / 1e3:,.0f}k"


def money_md(x: float) -> str:
    """money() for st.metric / markdown, where two "$" signs would be read as LaTeX."""
    return money(x).replace("$", "\\$")


def style_bars(fig, horizontal=True, pad=1.18):
    """Thinner bars, room for the value labels outside the bars."""
    fig.update_layout(bargap=0.45)
    values = [v for t in fig.data for v in (t.x if horizontal else t.y)]
    rng = [0, max(values) * pad]
    fig.update_layout(xaxis_range=rng) if horizontal else fig.update_layout(yaxis_range=rng)
    return fig


def page_header(title: str, caption: str):
    st.title(title)
    st.caption(caption)
