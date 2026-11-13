import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (BOROUGHS, BUILDING_CLASSES, HIGHLIGHT, HOME_TYPES, MUTED, area_unit, build_features,
                   card, chart, load_homes, load_model, load_results, money, page_header, takeaways, to_display_area, to_sqft)

homes = load_homes()
model = load_model()
test_pred = load_results("test_predictions")

page_header("Price predictor",
            "Estimate the 2025 sale price of a 1–3 family home with the final Gradient Boosting model")

# ---- Inputs ----
def n_sales(n):
    return f"{n} sale" if n == 1 else f"{n} sales"


form_col, result_col = st.columns([3, 2], gap="medium")

with form_col, card("inputs"):
    st.markdown("#### Describe the home")
    c1, c2, c3 = st.columns([2, 3, 2.4])
    borough = c1.selectbox("Borough", BOROUGHS, index=BOROUGHS.index("Queens"))
    in_borough = homes[homes["borough_name"] == borough]
    nb_counts = in_borough["neighborhood"].value_counts()
    nb_options = sorted(nb_counts.index)
    neighborhood = c2.selectbox("Neighbourhood", nb_options, index=nb_options.index(nb_counts.idxmax()),
                                format_func=lambda n: f"{n.title()} ({n_sales(nb_counts[n])})")
    zip_counts = in_borough.loc[in_borough["neighborhood"] == neighborhood, "zip_code"].value_counts()
    zip_code = c3.selectbox("ZIP code", zip_counts.index,
                            format_func=lambda z: f"{z} ({n_sales(zip_counts[z])})",
                            help="ZIP codes with 2025 sales in this neighbourhood, most sales first")

    c1, c2, c3 = st.columns([2, 3, 2.4])
    home_type = c1.selectbox("Home type", HOME_TYPES)

    class_counts = homes.loc[homes["home_type"] == home_type, "building_class_at_time_of_sale"].value_counts()
    building_class = c2.selectbox("Building class", class_counts.index,
                                  format_func=lambda c: f"{c}: {BUILDING_CLASSES.get(c, c)}")

    similar = homes[(homes["neighborhood"] == neighborhood) & (homes["home_type"] == home_type)]
    defaults = similar if len(similar) >= 3 else homes[homes["home_type"] == home_type]
    all_of_type = homes[homes["home_type"] == home_type]

    def default_value(col, lo, hi, convert=lambda v: v):
        """Median of the similar homes, falling back to all homes of this type when the
        neighbourhood has no recorded value (e.g. Breezy Point co-ops have no sizes),
        and kept inside the input's allowed range so the widget never errors."""
        source = defaults if defaults[col].notna().any() else all_of_type
        value = convert(source[col].median())
        return int(min(max(value, lo), hi))

    year_built = c3.number_input("Year built", 1800, 2025, default_value("year_built", 1800, 2025), step=1)

    # If none of the similar homes have a recorded size, start with "unknown" ticked
    sizes_missing_here = defaults["gross_square_feet"].isna().all()
    # Input limits per unit: the ft² limits cover the cleaned data, the m² limits are the same areas rounded
    unit = area_unit()
    gross_lo, gross_hi, land_lo, land_hi, step = ((300, 20_000, 200, 50_000, 50) if unit == "ft²"
                                                  else (25, 1_900, 20, 4_650, 5))
    c1, c2, c3 = st.columns(3)
    size_box = st.container()   # the checkbox is shown under the size inputs but must be read before them
    size_known = not size_box.checkbox(
        "Floor and land area unknown", value=bool(sizes_missing_here),
        help="The model then fills in typical sizes, as it does for missing values in the data")
    gross_in = c1.number_input(f"Gross floor area ({unit})", gross_lo, gross_hi,
                               default_value("gross_square_feet", gross_lo, gross_hi, to_display_area),
                               step=step, disabled=not size_known)
    land_in = c2.number_input(f"Land area ({unit})", land_lo, land_hi,
                              default_value("land_square_feet", land_lo, land_hi, to_display_area),
                              step=step, disabled=not size_known)
    # The model was trained on square feet, so convert back before predicting
    gross, land = to_sqft(gross_in), to_sqft(land_in)
    quarter = c3.selectbox("Quarter of sale (2025)", [1, 2, 3, 4], index=3, format_func=lambda q: f"Q{q}")

# ---- Prediction ----
X = build_features(borough, neighborhood, zip_code, home_type, building_class,
                   gross if size_known else None, land if size_known else None, year_built, quarter)
estimate = float(np.exp(model.predict(X)[0]))

# Realistic range from the model's errors on the unseen test set:
# actual = predicted / exp(residual), so residual quantiles turn into a price range
res = test_pred["residual_log"]
low50, high50 = estimate / np.exp(res.quantile([0.75, 0.25]))
low80, high80 = estimate / np.exp(res.quantile([0.90, 0.10]))


def position(price):
    """Where a price sits on the range bar (0–100%), on a log scale from the 80% low to the 80% high."""
    return 100 * (np.log(price) - np.log(low80)) / (np.log(high80) - np.log(low80))


with result_col:
    st.html(f"""
<div class="estimate">
  <div class="label">Estimated sale price</div>
  <div class="value">{money(estimate)}</div>
  <div class="sub">{home_type} home in {neighborhood.title()}, {borough} · Q{quarter} 2025</div>
  <div class="track">
    <div class="band" style="left: {position(low50):.1f}%; width: {position(high50) - position(low50):.1f}%"></div>
    <div class="marker" style="left: {position(estimate):.1f}%"></div>
  </div>
  <div class="ends"><span>{money(low80)}</span><span>{money(high80)}</span></div>
  <div class="row"><span>Likely range · 50% of sales</span><b>{money(low50)} – {money(high50)}</b></div>
  <div class="row"><span>Wide range · 80% of sales</span><b>{money(low80)} – {money(high80)}</b></div>
</div>""")
    st.caption("The ranges come from how far the model's predictions were from the real prices of 2,800 test-set "
               "sales it had never seen: half of those sales fell inside the 50% range, 80% inside the wide range.")

    if borough == "Manhattan":
        st.warning("Manhattan has very few 1–3 family sales (the model's median error there is about 20%), "
                   "so treat this estimate with extra caution.")
    if nb_counts[neighborhood] < 5:
        st.warning(f"Only {nb_counts[neighborhood]} sale(s) in {neighborhood.title()} in 2025, so the model knows "
                   "little about this neighbourhood's price level.")

# ---- What would change the estimate? The same home with one input changed at a time ----
st.subheader("What would change the estimate?")
inputs = dict(borough=borough, neighborhood=neighborhood, zip_code=zip_code, home_type=home_type,
              building_class=building_class, gross_sqft=gross if size_known else None,
              land_sqft=land if size_known else None, year_built=year_built, sale_quarter=quarter)
what_ifs = []   # (label, note, changed inputs)
if size_known:
    what_ifs.append(("20% more floor area", f"{gross_in * 1.2:,.0f} {unit} instead of {gross_in:,}",
                     {"gross_sqft": gross * 1.2}))
    what_ifs.append(("20% more land", f"{land_in * 1.2:,.0f} {unit} instead of {land_in:,}", {"land_sqft": land * 1.2}))
other_year = 2020 if year_built <= 2005 else 1960
what_ifs.append((f"Built in {other_year}", f"instead of {year_built}", {"year_built": other_year}))
# The same home in the most and the least expensive neighbourhood of the borough (at least 10 sales each)
nb_price = in_borough.groupby("neighborhood")["sale_price"].agg(["median", "size"])
nb_price = nb_price[(nb_price["size"] >= 10) & (nb_price.index != neighborhood)]
if len(nb_price) > 1:
    for nb, kind in [(nb_price["median"].idxmax(), "most"), (nb_price["median"].idxmin(), "least")]:
        nb_zip = in_borough.loc[in_borough["neighborhood"] == nb, "zip_code"].mode().iloc[0]
        what_ifs.append((f"In {nb.title()}", f"the {kind} expensive neighbourhood in {borough}",
                         {"neighborhood": nb, "zip_code": nb_zip}))

what_if_X = pd.concat([build_features(**{**inputs, **changed}) for _, _, changed in what_ifs], ignore_index=True)
what_if_price = np.exp(model.predict(what_if_X))
takeaways([(label, money(price), note, price / estimate - 1)
           for (label, note, _), price in zip(what_ifs, what_if_price)])
st.caption("Each card changes one input and asks the model again; the arrow shows the change from the estimate above. "
           "This shows how the model behaves. It is not a promise of what an extension or a move would add.")

# ---- Comparable sales ----
st.subheader("Comparable sales in the same neighbourhood")
comps = similar.dropna(subset=["gross_square_feet"]).copy()
if size_known and not comps.empty:
    comps["size_diff"] = (comps["gross_square_feet"] / gross - 1).abs()
    comps = comps[comps["size_diff"] <= 0.3].sort_values("size_diff")
comps = comps.head(10)

if comps.empty:
    st.info(f"No {home_type} sales of a similar size in {neighborhood.title()} in 2025.")
else:
    left, right = st.columns([3, 2])
    with left:
        comps["price_text"] = comps["sale_price"].map("${:,.0f}".format)
        comps[["gross_square_feet", "land_square_feet"]] = to_display_area(
            comps[["gross_square_feet", "land_square_feet"]]).round()
        st.dataframe(comps[["building_class_at_time_of_sale", "gross_square_feet", "land_square_feet",
                            "year_built", "sale_date", "price_text"]]
                     .rename(columns={"building_class_at_time_of_sale": "Class",
                                      "gross_square_feet": f"Floor area ({unit})",
                                      "land_square_feet": f"Land area ({unit})", "year_built": "Built",
                                      "sale_date": "Sold", "price_text": "Price"}),
                     hide_index=True, width="stretch",
                     column_config={"Sold": st.column_config.DateColumn(format="MMM YYYY"),
                                    "Built": st.column_config.NumberColumn(format="%d"),
                                    f"Floor area ({unit})": st.column_config.NumberColumn(format="localized"),
                                    f"Land area ({unit})": st.column_config.NumberColumn(format="localized")})
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comps["sale_price"], y=["Comparable sales"] * len(comps), mode="markers",
                                 marker=dict(size=12, color=MUTED, line=dict(width=2, color="white")),
                                 name="Comparable sale", hovertemplate="%{x:$,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[estimate], y=["Comparable sales"], mode="markers",
                                 marker=dict(size=18, color=HIGHLIGHT, symbol="diamond",
                                             line=dict(width=2, color="white")),
                                 name="Model estimate", hovertemplate="Estimate %{x:$,.0f}<extra></extra>"))
        fig.update_layout(height=260, title="Estimate vs comparable sales", xaxis_tickformat="$,.2s",
                          yaxis_visible=False, legend=dict(orientation="h", y=-0.35), margin=dict(t=50))
        chart(fig)
    st.caption("Comparables: same neighbourhood and home type, floor area within 30% of the home described "
               "above, closest size first.")

with st.expander("What the model receives"):
    st.dataframe(X, hide_index=True, width="stretch")
    st.caption("Sizes enter the model as the log of square feet, as in the notebook; areas typed in m² are "
               "converted to ft² first, so the estimate is the same in either unit.")
    st.caption("Sizes are log-transformed and building age is computed exactly as in the notebook. "
               "The saved pipeline then imputes, scales and encodes these values before the model predicts.")
