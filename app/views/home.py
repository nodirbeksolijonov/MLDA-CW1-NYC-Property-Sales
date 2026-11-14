import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import FINAL_MODEL_NAME, card, load_homes, load_results, load_zip_areas, money

homes = load_homes()
comparison = load_results("model_comparison")
final = comparison.loc[FINAL_MODEL_NAME]

@st.cache_data
def hero_map():
    """Small map for the hero: median 2025 sale price of each ZIP code area (lighter = more expensive)."""
    geo, zip_to_area = load_zip_areas()
    area = homes["zip_code"].map(zip_to_area)
    by_area = homes.groupby(area).agg(
        sales=("sale_price", "size"), median=("sale_price", "median"),
        place=("neighborhood_title", lambda n: n.value_counts().index[0]))
    shown = by_area[by_area["sales"] >= 5]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(   # every area in a muted blue underneath, so areas with few sales still show
        geojson=geo, locations=[f["id"] for f in geo["features"]], z=[0] * len(geo["features"]),
        colorscale=[[0, "#22406b"], [1, "#22406b"]], showscale=False, marker_line_color="#0f1e33",
        marker_line_width=0.4, hoverinfo="skip"))
    fig.add_trace(go.Choropleth(
        geojson=geo, locations=shown.index, z=np.log10(shown["median"]),
        zmin=np.log10(5e5), zmax=np.log10(6e6), showscale=False,
        colorscale=[[0, "#2f5f9e"], [0.45, "#6aa9f0"], [1, "#f2f7fd"]],
        marker_line_color="#0f1e33", marker_line_width=0.4,
        customdata=np.column_stack([shown["place"], shown["median"].map(money), shown["sales"]]),
        hovertemplate="%{customdata[0]}<br>Median %{customdata[1]} · %{customdata[2]} sales<extra></extra>"))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator")
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", dragmode=False,
                      hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#16202e", size=13)))
    return fig


with st.container(key="hero"):
    text_col, map_col = st.columns([6, 5], gap="large", vertical_alignment="center")
    with text_col:
        st.html(f"""
<div class="eyebrow">Machine Learning and Data Analytics coursework · Student 00017544</div>
<h1 class="hero-title">What is a New York home worth?</h1>
<p class="hero-lead"><b>{len(homes):,}</b> qualifying sales of one- to three-family homes in New York City in
2025, explored and priced by a machine learning model that is typically within <b>{final['test Median APE']:.1%}</b> of the real
sale price.</p>""")
        b1, b2 = st.columns([1, 1], gap="small")
        if b1.button("Estimate a price", type="primary", icon=":material/calculate:", width="stretch"):
            st.switch_page("views/predictor.py")
        if b2.button("Explore the market", icon=":material/monitoring:", width="stretch"):
            st.switch_page("views/dashboard.py")
    with map_col:
        st.plotly_chart(hero_map(), theme=None, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
        st.html('<p class="hero-map-note">Median 2025 sale price by ZIP code area · lighter = more expensive</p>')

st.markdown("""
Built on the public
[NYC Department of Finance Annualized Sales](https://www.nyc.gov/site/finance/property/property-annualized-sales-update.page)
records, cleaned and modelled in the accompanying Jupyter notebook.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Homes analysed", f"{len(homes):,}", help="Market sales after cleaning (non-market transfers removed)")
c2.metric("Median sale price", money(homes["sale_price"].median()))
c3.metric("Typical prediction error", f"{final['test Median APE']:.1%}",
          help="Median absolute percentage error of the final model on the unseen test set")
c4.metric("R² on log price", f"{final['test R² (log)']:.2f}",
          help="Share of the variance in log sale price explained on the unseen test set")

st.subheader("What you can do here")
SECTIONS = [
    ("dashboard", ":material/monitoring:", "Market dashboard",
     "Filter by borough, home type and quarter to see prices, sales volume and the most and least "
     "expensive neighbourhoods."),
    ("predictor", ":material/calculate:", "Price predictor",
     "Describe a home and get an estimated 2025 sale price, with a realistic range and comparable sales "
     "from the same neighbourhood."),
    ("performance", ":material/query_stats:", "Model performance",
     "How the four tuned models compare, how accurate the final model is on unseen sales, where it "
     "struggles and which features drive it."),
    ("data", ":material/description:", "Data & method",
     "Where the data comes from, how it was cleaned and what the model can and cannot do."),
]
for col, (page, icon, title, text) in zip(st.columns(4), SECTIONS):
    with col, card(f"nav_{page}", height="stretch"):
        st.markdown(f":blue[{icon}]")
        st.markdown(f"#### {title}")
        st.html(f'<p class="card-text">{text}</p>')
        st.page_link(f"views/{page}.py", label="Open", icon=":material/arrow_forward:")

st.subheader("How the model was built")
STEPS = [
    ("Data", f"85,068 sales records from the five borough files, narrowed to {len(homes):,} genuine "
             "1–3 family market sales."),
    ("Preparation", "Errors and non-market sales removed, features engineered (building age, log sizes, floor "
                    "area ratio), a stratified 70 / 15 / 15 train / validation / test split, and a leakage-free "
                    "preprocessing pipeline."),
    ("Models", "Linear Regression (Ridge), K-Nearest Neighbours, Random Forest and Histogram Gradient Boosting, "
               "each tuned with 5-fold cross-validated grid search."),
    ("Final model", f"<b>{FINAL_MODEL_NAME}</b>, re-trained on train + validation and evaluated once on the "
                    "test set."),
]
for i, (col, (title, text)) in enumerate(zip(st.columns(4), STEPS), start=1):
    with col, card(f"step_{i}", height="stretch"):
        st.html(f'<span class="card-icon">{i}</span>')
        st.markdown(f"#### {title}")
        st.html(f'<p class="step-text">{text}</p>')
st.page_link("views/lab.py", label="Train these models yourself, with your own settings, in the Model lab",
             icon=":material/science:")
