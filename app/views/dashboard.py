import numpy as np
import plotly.express as px
import streamlit as st

from utils import (BOROUGH_COLORS, BOROUGHS, HIGHLIGHT, HOME_TYPES, area_unit, load_homes, money, page_header,
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
    fig = px.bar(top, x="median", y="neighborhood_title", orientation="h", color="borough_name",
                 color_discrete_map=borough_colors, text=top["median"].map(money), custom_data=["sales"],
                 title=f"{order} neighbourhoods (at least {min_sales} sales)", labels={"borough_name": "Borough"})
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}<br>Median %{x:$,.0f}<br>%{customdata[0]:,} sales<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickformat="$,.2s",
                      yaxis={"categoryorder": "array", "categoryarray": top["neighborhood_title"].tolist()[::-1]})
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

with st.expander("Show the filtered data"):
    table = df[["borough_name", "neighborhood_title", "home_type", "building_class_at_time_of_sale",
                "gross_square_feet", "land_square_feet", "year_built", "sale_price", "sale_date"]].copy()
    table[["gross_square_feet", "land_square_feet"]] = to_display_area(
        table[["gross_square_feet", "land_square_feet"]]).round()
    st.dataframe(table.rename(columns={"gross_square_feet": f"floor area ({unit})",
                                       "land_square_feet": f"land area ({unit})"}),
                 width="stretch", hide_index=True)
