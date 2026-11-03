import streamlit as st

from utils import FINAL_MODEL_NAME, load_homes, load_results, money, page_header

homes = load_homes()
comparison = load_results("model_comparison")
final = comparison.loc[FINAL_MODEL_NAME]

page_header("NYC Home Prices 2025",
            "Machine Learning and Data Analytics coursework · Student 00017544")

st.markdown("""
This app explores **every 1–3 family home sold in New York City in 2025** and predicts the price of a home
from its location, size, age and type. It is built on the public
[NYC Department of Finance Annualized Sales](https://www.nyc.gov/site/finance/property/property-annualized-sales-update.page)
records, cleaned and modelled in the accompanying Jupyter notebook.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Homes analysed", f"{len(homes):,}", help="Market sales after cleaning (non-market transfers removed)")
c2.metric("Median sale price", money(homes["sale_price"].median()))
c3.metric("Typical prediction error", f"{final['test Median APE']:.1%}",
          help="Median absolute percentage error of the final model on the unseen test set")
c4.metric("Variance explained (R², log price)", f"{final['test R² (log)']:.2f}")

st.subheader("What you can do here")
a, b = st.columns(2)
with a:
    st.markdown("""
**📊 Market dashboard**: filter by borough, home type and quarter to see prices, sales volume
and the most and least expensive neighbourhoods.

**💲 Price predictor**: describe a home and get an estimated 2025 sale price, with a realistic range
and comparable sales from the same neighbourhood.
""")
with b:
    st.markdown("""
**🎯 Model performance**: how the four tuned models compare, how accurate the final model is on unseen
sales, where it struggles and which features drive its predictions.

**📄 Data & method**: where the data comes from, how it was cleaned and what the model can and cannot do.
""")

st.subheader("How the model was built")
st.markdown(f"""
1. **Data**: 85,068 sales records from the five borough files, narrowed to {len(homes):,} genuine 1–3 family market sales.
2. **Preparation**: errors and non-market sales removed, features engineered (building age, log sizes, floor area ratio),
   a stratified 70 / 15 / 15 train / validation / test split, and a leakage-free preprocessing pipeline.
3. **Models**: Linear Regression (Ridge), K-Nearest Neighbours, Random Forest and Histogram Gradient Boosting,
   each tuned with 5-fold cross-validated grid search.
4. **Final model**: **{FINAL_MODEL_NAME}**, re-trained on train + validation and evaluated once on the test set.
""")
