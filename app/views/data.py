import pandas as pd
import streamlit as st

from utils import load_homes, load_results, page_header

homes = load_homes()
log = load_results("cleaning_log").reset_index()

page_header("Data & method", "Where the data comes from, how it was prepared, and the limits of the model")

st.subheader("Data source")
st.markdown("""
- **NYC Department of Finance: Annualized Property Sales, 2025**, one Excel file per borough
  (Bronx, Brooklyn, Manhattan, Queens, Staten Island), 85,068 sales records in total.
  Source: [nyc.gov/site/finance](https://www.nyc.gov/site/finance/property/property-annualized-sales-update.page).
- The data is published by the City of New York as open data under the NYC Open Data Terms of Use.
- Each record is one sale: location (borough, neighbourhood, block/lot, address), building class,
  units, land and floor area, year built, sale price and sale date.
""")

st.subheader("From raw records to the modelling dataset")
st.markdown("""
The analysis keeps **1–3 family homes sold for at least \\$10,000**. About half of all 1–3 family records are
transfers for \\$0 or a nominal amount (e.g. between family members), which are not market sales.
The remaining cleaning steps:
""")
log["removed"] = (-log["rows"].diff()).fillna(0).astype(int)
st.dataframe(log, hide_index=True, width="stretch",
             column_config={"step": "Cleaning step",
                            "rows": st.column_config.NumberColumn("Rows remaining", format="localized"),
                            "removed": st.column_config.NumberColumn("Removed", format="localized")})
st.markdown("""
- **Removed sales whose building class changed after the sale** (demolished, converted or rebuilt, so the
  recorded features no longer describe what was sold).
- **Removed multi-lot package sales**: several lots sold together for one total price, recorded on every lot.
- **Removed very low price-per-square-foot sales** within each borough (lower IQR fence on the log scale):
  these are almost always non-market sales. Very high values were checked and kept, since they are real
  luxury homes in premium neighbourhoods.
- Corrected impossible unit counts and floor areas, and dropped empty, constant, identifier and post-sale columns.
""")

st.subheader("Features used by the model")
st.dataframe(pd.DataFrame([
    ("Neighbourhood", "Target-encoded (cross-fitted) average log price of the neighbourhood"),
    ("ZIP code", "Target-encoded in the same way; refines the location inside a neighbourhood"),
    ("Borough", "One-hot encoded"),
    ("Building class", "One-hot encoded (rare classes grouped)"),
    ("Home type", "Ordinal: 1-family < 2-family < 3-family"),
    ("Gross floor area, land area", "Log-transformed, median-imputed, with a missing-size flag, scaled"),
    ("Floor area ratio", "Floor area / land area, scaled"),
    ("Building age", "2025 − year built, scaled"),
    ("Has a shop/office", "Yes/no, from the commercial units"),
    ("Sale quarter", "One-hot encoded"),
], columns=["Feature", "How it is prepared"]), hide_index=True, width="stretch")
st.markdown("""
The target is the **log of the sale price**, because prices are strongly right-skewed. Predictions are converted
back to dollars for all reported errors. All preparation steps are fitted on the training data only, inside one
scikit-learn pipeline, so no information from the validation or test sets leaks into the model.
""")

st.subheader("Limitations")
st.markdown(f"""
- The public record has **no information on condition, renovation, views or exact position** below
  ZIP-code level, which is why a typical error of about 14% remains.
- **Manhattan** has very few 1–3 family sales ({(homes['borough_name'] == 'Manhattan').sum()} after cleaning),
  so predictions there are much less reliable.
- The model reflects **2025 prices** and would need re-training on newer sales as the market changes.
- Some below-market sales cannot be identified from the data and remain in it; the model over-prices them.
""")
