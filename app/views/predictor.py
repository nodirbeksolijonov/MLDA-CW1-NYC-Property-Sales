import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (BOROUGHS, BUILDING_CLASSES, HIGHLIGHT, HOME_TYPES, MUTED, SQFT_PER_M2, area_unit, build_features,
                   card, chart, load_homes, load_model, load_results, money, page_header, palette, seed, set_url, takeaways,
                   to_display_area, to_sqft, url_value)

homes = load_homes()
model = load_model()
# Errors behind the price ranges: out-of-fold cross-validation errors on the training and validation homes,
# and the share of test sales that fell inside each range (notebook Section 6.2)
cv_res = load_results("cv_residuals")["residual_log"]
range_check = load_results("price_ranges")["share of test sales inside"]

page_header("Price predictor",
            "Estimate the 2025 sale price of a 1–3 family home with the final Gradient Boosting model")

# ---- Inputs ----
def n_sales(n):
    return f"{n} sale" if n == 1 else f"{n} sales"


form_col, result_col = st.columns([3, 2], gap="medium")

with form_col, card("inputs"):
    st.markdown("#### Describe the home")
    # Every input starts from the page address when a shared link carries it, otherwise from a sensible default.
    # An input whose choices depend on another input has that input in its key, so it starts afresh when it changes.
    c1, c2, c3 = st.columns([2, 3, 2.4])
    seed("pred_borough", url_value("borough", BOROUGHS) or "Queens")
    borough = c1.selectbox("Borough", BOROUGHS, key="pred_borough")
    in_borough = homes[homes["borough_name"] == borough]
    nb_counts = in_borough["neighborhood"].value_counts()
    nb_options = sorted(nb_counts.index)
    nb_default = nb_counts.idxmax()
    seed(f"pred_nb_{borough}", url_value("nb", nb_options) or nb_default)
    neighborhood = c2.selectbox("Neighbourhood", nb_options, key=f"pred_nb_{borough}",
                                format_func=lambda n: f"{n.title()} ({n_sales(nb_counts[n])})")
    zip_counts = in_borough.loc[in_borough["neighborhood"] == neighborhood, "zip_code"].value_counts()
    zip_options = list(zip_counts.index)
    seed(f"pred_zip_{borough}_{neighborhood}", url_value("zip", zip_options) or zip_options[0])
    zip_code = c3.selectbox("ZIP code", zip_options, key=f"pred_zip_{borough}_{neighborhood}",
                            format_func=lambda z: f"{z} ({n_sales(zip_counts[z])})",
                            help="ZIP codes with 2025 sales in this neighbourhood, most sales first")

    c1, c2, c3 = st.columns([2, 3, 2.4])
    seed("pred_type", url_value("type", HOME_TYPES) or HOME_TYPES[0])
    home_type = c1.selectbox("Home type", HOME_TYPES, key="pred_type")

    class_counts = homes.loc[homes["home_type"] == home_type, "building_class_at_time_of_sale"].value_counts()
    class_options = list(class_counts.index)
    seed(f"pred_class_{home_type}", url_value("class", class_options) or class_options[0])
    building_class = c2.selectbox("Building class", class_options, key=f"pred_class_{home_type}",
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

    # The typical year and sizes depend on the neighbourhood and home type, so these inputs restart with them
    context = f"{borough}_{neighborhood}_{home_type}"
    year_default = default_value("year_built", 1800, 2025)
    seed(f"pred_year_{context}", url_value("year", range(1800, 2026), int) or year_default)
    year_built = c3.number_input("Year built", 1800, 2025, key=f"pred_year_{context}", step=1)

    # If none of the similar homes have a recorded size, start with "unknown" ticked
    sizes_missing_here = defaults["gross_square_feet"].isna().all()
    # Input limits per unit: the ft² limits cover the cleaned data, the m² limits are the same areas rounded
    unit = area_unit()
    gross_lo, gross_hi, land_lo, land_hi, step = ((300, 20_000, 200, 50_000, 50) if unit == "ft²"
                                                  else (25, 1_900, 20, 4_650, 5))
    c1, c2, c3 = st.columns(3)
    size_box = st.container()   # the checkbox is shown under the size inputs but must be read before them
    unknown_default = bool(sizes_missing_here)
    seed(f"pred_unknown_{context}", {"1": True, "0": False}.get(st.query_params.get("unknown"), unknown_default))
    size_known = not size_box.checkbox(
        "Floor and land area unknown", key=f"pred_unknown_{context}",
        help="The model then fills in typical sizes, as it does for missing values in the data")
    gross_default = default_value("gross_square_feet", gross_lo, gross_hi, to_display_area)
    land_default = default_value("land_square_feet", land_lo, land_hi, to_display_area)
    # When the area unit is switched, the sizes already typed are carried over (converted), not reset to the
    # defaults. `restore` holds, per size, (value shown, value it was converted from, whether it was rounded), so switching
    # back returns the original number exactly.
    size_limits = {"gross": (gross_lo, gross_hi), "land": (land_lo, land_hi)}
    memo = st.session_state.get(f"pred_sizes_{context}")
    restore = {}
    if memo and memo["unit"] == unit:
        restore = memo["restore"]
    elif memo and f"pred_gross_{context}_{memo['unit']}" in st.session_state:
        for name, (lo, hi) in size_limits.items():
            typed = memo[name]
            shown, original, was_rounded = memo["restore"].get(name, (None, None, False))
            if typed == shown:   # unchanged since the last switch: return the value it came from
                converted, rounded = original, not was_rounded
            else:
                converted, rounded = round(typed / SQFT_PER_M2 if unit == "m²" else typed * SQFT_PER_M2), True
            converted = int(min(max(converted, lo), hi))
            st.session_state[f"pred_{name}_{context}_{unit}"] = converted
            restore[name] = (converted, typed, rounded)
    # sizes in a shared link are in the link's unit, so they are only used when that is the unit shown
    same_unit = (st.query_params.get("unit") == "m2") == (unit == "m²")
    seed(f"pred_gross_{context}_{unit}",
         (url_value("floor", range(gross_lo, gross_hi + 1), int) if same_unit else None) or gross_default)
    seed(f"pred_land_{context}_{unit}",
         (url_value("land", range(land_lo, land_hi + 1), int) if same_unit else None) or land_default)
    gross_in = c1.number_input(f"Gross floor area ({unit})", gross_lo, gross_hi, key=f"pred_gross_{context}_{unit}",
                               step=step, disabled=not size_known)
    land_in = c2.number_input(f"Land area ({unit})", land_lo, land_hi, key=f"pred_land_{context}_{unit}",
                              step=step, disabled=not size_known)
    st.session_state[f"pred_sizes_{context}"] = {"unit": unit, "gross": gross_in, "land": land_in, "restore": restore}
    # The model was trained on square feet, so convert back before predicting

    def model_sqft(name, typed):
        """ft² for the model. A size carried over from the other unit is shown rounded (3,000 ft² as 279 m²),
        so the value it came from is used and the estimate does not move when only the unit is switched."""
        shown, original, rounded = restore.get(name, (None, None, False))
        lo, hi = size_limits[name]
        # typed here, edited since the switch, or cut off at the input's limit: use the number as shown
        if not rounded or typed != shown or typed in (lo, hi):
            return to_sqft(typed)
        return original if unit == "m²" else original * SQFT_PER_M2

    gross, land = model_sqft("gross", gross_in), model_sqft("land", land_in)
    seed("pred_quarter", url_value("q", [1, 2, 3, 4], int) or 4)
    quarter = c3.selectbox("Quarter of sale (2025)", [1, 2, 3, 4], key="pred_quarter", format_func=lambda q: f"Q{q}")

# Keep the page address in step with the inputs (only what differs from the defaults), so it can be shared
set_url(**{
    "borough": borough if borough != "Queens" else None,
    "nb": neighborhood if neighborhood != nb_default else None,
    "zip": zip_code if zip_code != zip_options[0] else None,
    "type": home_type if home_type != HOME_TYPES[0] else None,
    "class": building_class if building_class != class_options[0] else None,
    "year": int(year_built) if year_built != year_default else None,
    "floor": int(gross_in) if size_known and gross_in != gross_default else None,
    "land": int(land_in) if size_known and land_in != land_default else None,
    "unknown": ("0" if size_known else "1") if size_known == unknown_default else None,
    "q": quarter if quarter != 4 else None,
})
form_col.caption(":material/link: The page address follows your inputs, so you can copy it to share this exact estimate.")

# ---- Prediction ----
X = build_features(borough, neighborhood, zip_code, home_type, building_class,
                   gross if size_known else None, land if size_known else None, year_built, quarter)
estimate = float(np.exp(model.predict(X)[0]))

# Realistic range from the model's cross-validation errors (the test set is used only to check it):
# actual = predicted / exp(residual), so residual quantiles turn into a price range
low50, high50 = estimate / np.exp(cv_res.quantile([0.75, 0.25]))
low80, high80 = estimate / np.exp(cv_res.quantile([0.90, 0.10]))


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
    st.caption(f"The ranges come from the model's cross-validation errors on the {len(cv_res):,} training and "
               f"validation sales, each predicted by a model that had not seen it. Checked once on test sales it had "
               f"never seen: "
               f"{range_check['50% range']:.0%} fell inside the 50% range and {range_check['80% range']:.0%} inside "
               f"the wide range.")

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
else:   # no size to compare with, so show the most recent sales instead
    comps = comps.sort_values("sale_date", ascending=False)
comps = comps.head(10)

if comps.empty:
    st.info(f"No {home_type} sales {'of a similar size ' if size_known else 'with a recorded size '}"
            f"in {neighborhood.title()} in 2025.")
else:
    left, right = st.columns([3, 2])
    with left:
        comps["price_text"] = comps["sale_price"].map("${:,.0f}".format)
        comps[["gross_square_feet", "land_square_feet"]] = to_display_area(
            comps[["gross_square_feet", "land_square_feet"]]).round()
        comps_table = (comps[["building_class_at_time_of_sale", "gross_square_feet", "land_square_feet",
                              "year_built", "sale_date", "price_text"]]
                       .rename(columns={"building_class_at_time_of_sale": "Class",
                                        "gross_square_feet": f"Floor area ({unit})",
                                        "land_square_feet": f"Land area ({unit})", "year_built": "Built",
                                        "sale_date": "Sold", "price_text": "Price"}))
        st.dataframe(comps_table, hide_index=True, width="stretch",
                     column_config={"Sold": st.column_config.DateColumn(format="MMM YYYY"),
                                    "Built": st.column_config.NumberColumn(format="%d"),
                                    f"Floor area ({unit})": st.column_config.NumberColumn(format="localized"),
                                    f"Land area ({unit})": st.column_config.NumberColumn(format="localized")})
        st.download_button("Download as CSV", on_click="ignore", icon=":material/download:",
                           data=comps_table.assign(Price=comps["sale_price"]).to_csv(index=False).encode("utf-8"),
                           file_name=f"comparable_sales_{neighborhood.lower().replace(' ', '_')}.csv", mime="text/csv")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comps["sale_price"], y=["Comparable sales"] * len(comps), mode="markers",
                                 marker=dict(size=12, color=MUTED, line=dict(width=2, color=palette()["card"])),
                                 name="Comparable sale", hovertemplate="%{x:$,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[estimate], y=["Comparable sales"], mode="markers",
                                 marker=dict(size=18, color=HIGHLIGHT, symbol="diamond",
                                             line=dict(width=2, color=palette()["card"])),
                                 name="Model estimate", hovertemplate="Estimate %{x:$,.0f}<extra></extra>"))
        fig.update_layout(height=260, title="Estimate vs comparable sales", xaxis_tickformat="$,.2s",
                          yaxis_visible=False, legend=dict(orientation="h", y=-0.35), margin=dict(t=50))
        chart(fig)
    if size_known:
        st.caption("Comparables: same neighbourhood and home type, floor area within 30% of the home described "
                   "above, closest size first.")
    else:
        st.caption("Comparables: same neighbourhood and home type. The size is unknown, so sales are not "
                   "filtered by size; the 10 most recent sales with a recorded size are shown.")

with st.expander("What the model receives"):
    st.dataframe(X, hide_index=True, width="stretch")
    st.caption("Sizes enter the model as the log of square feet, as in the notebook; areas typed in m² are "
               "converted to ft² first, so the estimate is the same in either unit.")
    st.caption("Sizes are log-transformed and building age is computed exactly as in the notebook. "
               "The saved pipeline then imputes, scales and encodes these values before the model predicts.")
