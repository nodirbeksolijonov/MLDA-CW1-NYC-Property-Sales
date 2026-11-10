import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (BOROUGH_COLORS, BOROUGHS, HIGHLIGHT, HOME_TYPES, MUTED, area_unit, load_homes, money, page_header,
                   style_bars, to_display_area)

homes = load_homes()
page_header("Market dashboard", "1–3 family market sales in New York City, January–December 2025")

# ---- Filters, in one row above the charts ----
f1, f2, f3 = st.columns([3, 2, 2])
boroughs = f1.multiselect("Borough", BOROUGHS, default=BOROUGHS)
types = f2.multiselect("Home type", HOME_TYPES, default=HOME_TYPES)
q_from, q_to = f3.select_slider("Quarter", options=[1, 2, 3, 4], value=(1, 4), format_func=lambda q: f"Q{q}")

df = homes[homes["borough_name"].isin(boroughs) & homes["home_type"].isin(types)
           & homes["sale_quarter"].between(q_from, q_to)]
if df.empty:
    st.warning("No sales match these filters.")
    st.stop()

unit = area_unit()
floor_area = to_display_area(df["gross_square_feet"])
price_per_area = df["sale_price"] / floor_area
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sales", f"{len(df):,}")
k2.metric("Median price", money(df["sale_price"].median()))
k3.metric(f"Median price per {unit}", f"${price_per_area.median():,.0f}")
k4.metric("Median building size", f"{floor_area.median():,.0f} {unit}")

PLOT = dict(theme="streamlit", width="stretch")
borough_colors = {b: BOROUGH_COLORS[b] for b in boroughs}

# ---- Row 1: price by borough, price distribution ----
left, right = st.columns(2)
with left:
    by_b = (df.groupby("borough_name")["sale_price"].agg(median="median", sales="size")
            .reindex([b for b in BOROUGHS if b in boroughs]).dropna().sort_values("median"))
    fig = px.bar(by_b.reset_index(), x="median", y="borough_name", orientation="h", color="borough_name",
                 color_discrete_map=borough_colors, text=by_b["median"].map(money),
                 custom_data=["sales"], title="Median sale price by borough")
    fig.update_traces(hovertemplate="%{y}<br>Median %{x:$,.0f}<br>%{customdata[0]:,} sales<extra></extra>",
                      textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, xaxis_tickformat="$,.2s")
    style_bars(fig)
    st.plotly_chart(fig, **PLOT)
with right:
    # histogram of log10(price), with the axis labelled in dollars
    fig = px.histogram(x=np.log10(df["sale_price"]), nbins=50, title="Distribution of sale prices (log scale)",
                       color_discrete_sequence=[HIGHLIGHT])
    ticks = [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7]
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="%{y:,} sales<extra></extra>")
    fig.update_layout(xaxis=dict(title="Sale price", tickvals=np.log10(ticks), ticktext=[money(t) for t in ticks]),
                      yaxis_title="Sales", bargap=0)
    st.plotly_chart(fig, **PLOT)

# ---- Row 2: monthly trend (two separate charts, never a dual axis) ----
monthly = (df.groupby(["sale_month", "borough_name"])["sale_price"]
           .agg(median="median", sales="size").reset_index())
left, right = st.columns(2)
with left:
    fig = px.line(monthly, x="sale_month", y="median", color="borough_name", markers=True,
                  color_discrete_map=borough_colors, category_orders={"borough_name": BOROUGHS},
                  title="Median price by month", labels={"borough_name": "Borough"})
    fig.update_traces(line_width=2, marker_size=8,
                      hovertemplate="%{x|%b %Y}<br>Median %{y:$,.0f}<extra>%{fullData.name}</extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_tickformat="$,.2s", hovermode="x unified")
    st.plotly_chart(fig, **PLOT)
with right:
    fig = px.bar(monthly, x="sale_month", y="sales", color="borough_name", color_discrete_map=borough_colors,
                 category_orders={"borough_name": BOROUGHS}, title="Number of sales by month",
                 labels={"borough_name": "Borough"})
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="%{x|%b %Y}<br>%{y:,} sales<extra>%{fullData.name}</extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, bargap=0.25)
    st.plotly_chart(fig, **PLOT)
if df["borough_name"].eq("Manhattan").any():
    st.caption("Manhattan has very few 1–3 family sales per month, so its monthly median jumps around a lot.")

# ---- Row 3: neighbourhoods and size vs price ----
left, right = st.columns(2)
with left:
    min_sales = 10
    nb = (df.groupby(["neighborhood_title", "borough_name"])["sale_price"]
          .agg(median="median", sales="size").reset_index().query("sales >= @min_sales"))
    order = st.radio("Neighbourhoods", ["Most expensive", "Least expensive"], horizontal=True)
    top = nb.nlargest(10, "median") if order == "Most expensive" else nb.nsmallest(10, "median")
    top = top.sort_values("median", ascending=order != "Most expensive")
    if top.empty:   # e.g. Staten Island 3-family: no neighbourhood reaches the minimum number of sales
        st.info(f"No neighbourhood has at least {min_sales} sales with these filters.")
    else:
        fig = px.bar(top, x="median", y="neighborhood_title", orientation="h", color="borough_name",
                     color_discrete_map=borough_colors, text=top["median"].map(money), custom_data=["sales"],
                     title=f"{order} neighbourhoods (at least {min_sales} sales)",
                     labels={"borough_name": "Borough"})
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>Median %{x:$,.0f}<br>%{customdata[0]:,} sales<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickformat="$,.2s",
                          yaxis={"categoryorder": "array",
                                 "categoryarray": top["neighborhood_title"].tolist()[::-1]})
        style_bars(fig, pad=1.25)
        st.plotly_chart(fig, **PLOT)
with right:
    sample = df.dropna(subset=["gross_square_feet"])
    sample = sample.sample(min(len(sample), 3000), random_state=42)
    sample = sample.assign(floor_area=to_display_area(sample["gross_square_feet"]))
    fig = px.scatter(sample, x="floor_area", y="sale_price", log_x=True, log_y=True, opacity=0.4,
                     color_discrete_sequence=[HIGHLIGHT], custom_data=["neighborhood_title", "borough_name"],
                     title="Building size vs sale price (log scales)")
    fig.update_traces(marker_size=8, marker_line_width=0, hovertemplate=(
        "%{customdata[0]}, %{customdata[1]}<br>%{x:,.0f} "
        f"{unit}<br>%{{y:$,.0f}}<extra></extra>"))
    price_ticks = [2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7]
    size_ticks = [500, 1000, 2000, 5000, 10000] if unit == "ft²" else [50, 100, 200, 500, 1000]
    fig.update_layout(xaxis=dict(title=f"Gross floor area ({unit})", tickvals=size_ticks,
                                 ticktext=[f"{t:,}" for t in size_ticks]),
                      yaxis=dict(title="Sale price", tickvals=price_ticks, ticktext=[money(t) for t in price_ticks]))
    st.plotly_chart(fig, **PLOT)
    st.caption(f"Random sample of {len(sample):,} sales. Bigger homes cost more, but at any size the price "
               "still varies a lot, mostly because of location.")

# ---- Row 4: year built (price and number of homes on separate panels, never a dual axis) ----
built = df.dropna(subset=["year_built"])
decade = (built["year_built"] // 10 * 10).clip(lower=1880).astype(int)   # the few pre-1890 homes form one group
by_decade = built.groupby(decade)["sale_price"].agg(median="median", sales="size")
by_decade.index = ["before 1890" if d == 1880 else f"{d}s" for d in by_decade.index]
MIN_DECADE_SALES = 10   # a median of fewer sales is too noisy to plot
reliable = by_decade[by_decade["sales"] >= MIN_DECADE_SALES]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
fig.add_trace(go.Scatter(x=reliable.index, y=reliable["median"], mode="lines+markers",
                         line=dict(width=2, color=HIGHLIGHT), marker=dict(size=8, color=HIGHLIGHT),
                         customdata=reliable["sales"],
                         hovertemplate="Built %{x}<br>Median %{y:$,.0f}<br>%{customdata:,} sales<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Bar(x=by_decade.index, y=by_decade["sales"], marker_color=MUTED, marker_line_width=0,
                     hovertemplate="Built %{x}<br>%{y:,} sales<extra></extra>"), row=2, col=1)
fig.update_layout(title="Sale price by decade built", showlegend=False, height=460, bargap=0.25,
                  hovermode="x unified")
fig.update_yaxes(title_text="Median price", tickformat="$,.2s", row=1, col=1)
fig.update_yaxes(title_text="Sales", row=2, col=1)
fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(by_decade.index))
st.plotly_chart(fig, **PLOT)
st.caption(f"Price follows a U-shape: pre-war and new homes sell for more, homes built around 1975–2000 for the "
           f"least. Much of this is location: about two-thirds of homes built before 1910 are in Brooklyn, and "
           f"two-thirds of those built 1975–2000 are in suburban Staten Island. Most recorded build years are rounded estimates "
           f"(about 78% end in 0 or 5, and many pre-1900 homes are recorded as 1899), so decades are used. "
           f"Prices are shown only for decades with at least {MIN_DECADE_SALES} sales.")

# ---- Row 5: correlation heatmap of the model's numeric features ----
CORR_FEATURES = {   # column in the cleaned data -> label; the same features the model uses (notebook 4b)
    "log_price": "Log price", "log_floor": "Log floor area", "log_land": "Log land area",
    "floor_area_ratio": "Floor area ratio", "building_age": "Building age", "home_type_n": "Home type (1–3)",
    "has_commercial": "Has a shop/office", "sale_quarter": "Sale quarter",
}


@st.cache_data
def correlation_matrix(features: pd.DataFrame, method: str) -> pd.DataFrame:
    """Cached on the filtered data and the method, so switching the toggle stays fast."""
    return features.corr(method=method.lower())


features = pd.DataFrame({
    "log_price": np.log(df["sale_price"]),
    "log_floor": np.log(df["gross_square_feet"]),
    "log_land": np.log(df["land_square_feet"]),
    "floor_area_ratio": df["floor_area_ratio"],
    "building_age": df["building_age"],
    "home_type_n": df["home_type"].str[0].astype(int),
    "has_commercial": df["has_commercial"],
    "sale_quarter": df["sale_quarter"],
}).rename(columns=CORR_FEATURES)
# A feature with a single value under the current filters has no correlation (e.g. no shops among
# Manhattan 3-family homes), so it is left out and named in a note instead of showing an empty row
constant = [c for c in features.columns if features[c].nunique() <= 1]
features = features.drop(columns=constant)

st.subheader("How the model's numeric features relate to each other")
method = st.radio("Correlation", ["Spearman", "Pearson"], horizontal=True,
                  help="Spearman uses ranks, so it captures any steadily rising or falling relationship and is "
                       "less affected by outliers; Pearson measures straight-line relationships only")
corr = correlation_matrix(features, method)
# Show the lower triangle only: the upper half repeats it and the diagonal is always 1
corr_lower = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1)).iloc[1:, :-1]
fig = go.Figure(go.Heatmap(
    z=corr_lower.values, x=corr_lower.columns, y=corr_lower.index, zmin=-1, zmax=1, zmid=0,
    colorscale=[[0, "#2a78d6"], [0.5, "#f1f0ed"], [1, "#d6542a"]],   # blue = negative, grey = 0, red = positive
    text=corr_lower.round(2).values, texttemplate="%{text:.2f}", textfont=dict(color="#1f1f1d", size=13),
    xgap=2, ygap=2, hoverongaps=False, colorbar=dict(title=method, thickness=12),
    hovertemplate="%{y} vs %{x}<br>" + method + " correlation %{z:.2f}<extra></extra>"))
fig.update_layout(height=480, title=f"{method} correlation ({len(df):,} sales with the current filters)",
                  xaxis=dict(tickangle=-35, showgrid=False), yaxis=dict(autorange="reversed", showgrid=False),
                  plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, **PLOT)
st.caption("Neighbourhood, ZIP code and borough are not shown because they are categories, not numbers, yet "
           "location is the strongest driver of price (see the feature importance on the Model performance page). "
           "Try filtering by borough: the link between size and price differs between boroughs. Homes with no "
           "recorded size are left out of the pairs that use size."
           + (f" Not shown because they have only one value with these filters: {', '.join(constant)}."
              if constant else ""))

with st.expander("Show the filtered data"):
    table = df[["borough_name", "neighborhood_title", "home_type", "building_class_at_time_of_sale",
                "gross_square_feet", "land_square_feet", "year_built", "sale_price", "sale_date"]].copy()
    table[["gross_square_feet", "land_square_feet"]] = to_display_area(
        table[["gross_square_feet", "land_square_feet"]]).round()
    st.dataframe(table.rename(columns={"gross_square_feet": f"floor area ({unit})",
                                       "land_square_feet": f"land area ({unit})"}),
                 width="stretch", hide_index=True)
