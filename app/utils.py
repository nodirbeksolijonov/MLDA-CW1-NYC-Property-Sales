"""Shared loading, feature engineering and styling for the Streamlit app.

The feature engineering in `build_features` must stay identical to Section 4b
of the notebook, because the saved pipeline expects exactly those columns.
"""
import json
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
    "zip_code": "ZIP code", "borough_name": "Borough", "building_age": "Building age", "home_type": "Home type",
    "building_class_at_time_of_sale": "Building class", "floor_area_ratio": "Floor area ratio",
    "sale_quarter": "Sale quarter", "has_commercial": "Has a shop/office",
}


SQFT_PER_M2 = 10.7639   # 1 m² = 10.7639 ft²


def area_unit() -> str:
    """The unit chosen in the sidebar ("ft²" or "m²"). The data and the model stay in ft²;
    only what is displayed or typed in is converted."""
    return st.session_state.get("area_unit", "ft²")


def to_display_area(sqft):
    """Convert an area in ft² (number or Series) to the chosen display unit."""
    return sqft / SQFT_PER_M2 if area_unit() == "m²" else sqft


def to_sqft(value):
    """Convert an area typed in the chosen display unit back to ft² for the model."""
    return value * SQFT_PER_M2 if area_unit() == "m²" else value


@st.cache_data
def load_homes() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, index_col="row_id", parse_dates=["sale_date"])
    df["sale_month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()
    df["neighborhood_title"] = df["neighborhood"].str.title()
    df["zip_code"] = df["zip_code"].astype(str)   # a label, as in the notebook
    return df


ZIP_GEO_FILE = ROOT / "data" / "nyc_zip_boundaries.geojson"


@st.cache_data
def load_zip_areas():
    """ZIP boundary areas (NYC Open Data MODZCTA, simplified) and a ZIP code -> area lookup.
    A few areas cover more than one ZIP code, so sales are grouped by area."""
    geo = json.loads(ZIP_GEO_FILE.read_text())
    zip_to_area = {z: f["id"] for f in geo["features"] for z in f["properties"]["covers"]}
    return geo, zip_to_area


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_results(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{name}.csv", index_col=0)


def build_features(borough, neighborhood, zip_code, home_type, building_class,
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
        "zip_code": str(zip_code),
    }])


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x / 1e3:,.0f}k"


def money_md(x: float) -> str:
    """money() for st.metric / markdown, where two "$" signs would be read as LaTeX."""
    return money(x).replace("$", "\\$")


def style_bars(fig, horizontal=True, pad=1.3):
    """Thinner bars, room for the value labels outside the bars."""
    fig.update_layout(bargap=0.45)
    values = [v for t in fig.data for v in (t.x if horizontal else t.y)]
    rng = [0, max(values) * pad]
    fig.update_layout(xaxis_range=rng) if horizontal else fig.update_layout(yaxis_range=rng)
    return fig


ASSETS = Path(__file__).resolve().parent / "assets"


def apply_style():
    """Load the shared stylesheet (app/assets/style.css). Called once per run from streamlit_app.py."""
    st.html(f"<style>{(ASSETS / 'style.css').read_text()}</style>")
    st.session_state["_chart_count"] = 0


def page_header(title: str, caption: str, eyebrow: str = "NYC home prices 2025"):
    st.html(f'<div class="eyebrow">{eyebrow}</div>')
    st.title(title)
    st.html(f'<p class="page-lead">{caption}</p>')


def card(key: str, **kwargs):
    """A white bordered card; the key gives it the st-key-card_... class styled in style.css."""
    return st.container(border=True, key=f"card_{key}", **kwargs)


def takeaways(items):
    """A row of small insight cards. Each item is (label, value, note) or (label, value, note, change),
    where change is a fraction such as 0.12 and is shown as an up or down arrow with the percentage."""
    cards = []
    for label, value, note, *change in items:
        delta = ""
        if change and abs(change[0]) < 0.005:
            delta = '<span class="t-delta">No change</span> · '
        elif change:
            up = change[0] > 0
            delta = (f'<span class="t-delta {"up" if up else "down"}">{"▲" if up else "▼"} '
                     f'{change[0]:+.0%}</span> ')
        cards.append(f'<div class="takeaway"><div class="t-label">{label}</div>'
                     f'<div class="t-value">{value}</div><div class="t-note">{delta}{note}</div></div>')
    st.html(f'<div class="takeaways">{"".join(cards)}</div>')


REPO_URL = "https://github.com/nodirbeksolijonov/MLDA-CW1-NYC-Property-Sales"


def page_footer():
    """Shown under every page, from streamlit_app.py."""
    st.html(f'''<div class="footer">
  <span>NYC Home Prices 2025 · Machine Learning and Data Analytics coursework · Student 00017544</span>
  <span>Data: <a href="https://www.nyc.gov/site/finance/property/property-annualized-sales-update.page"
  target="_blank">NYC Department of Finance</a> · <a href="{REPO_URL}" target="_blank">Source code</a></span>
</div>''')


def chart(fig):
    """Show a Plotly figure on a white card, in the app theme and without the Plotly toolbar."""
    fig.update_layout(hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dfe3ea", font=dict(color="#16202e", size=13)))
    if fig.layout.title.text:   # setting a title font on a chart without a title would show "undefined"
        fig.update_layout(title_font=dict(size=15, color="#16202e"))
    st.session_state["_chart_count"] += 1
    with st.container(border=True, key=f"chart_{st.session_state['_chart_count']}"):
        st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False})
