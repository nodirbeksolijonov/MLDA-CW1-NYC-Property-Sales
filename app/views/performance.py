import ast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (BOROUGHS, FEATURE_LABELS, FINAL_MODEL_NAME, HIGHLIGHT, MUTED, chart, load_results, money,
                   page_header, palette, set_url, style_bars, takeaways, url_value)

comparison = load_results("model_comparison")
test_pred = load_results("test_predictions")
importance = load_results("permutation_importance")
final = comparison.loc[FINAL_MODEL_NAME]

page_header("Model performance",
            f"Four tuned models compared, and the final {FINAL_MODEL_NAME} model evaluated on 2,800 unseen test sales")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Median error (test)", f"{final['test Median APE']:.1%}")
k2.metric("Mean absolute error (test)", money(final["test MAE ($)"]))
k3.metric("R² on log price (test)", f"{final['test R² (log)']:.3f}")
k4.metric("RMSE on log price (test)", f"{final['test RMSE (log)']:.3f}")

# ---- The four parts of the page, one function each. They are grouped into tabs at the end of the file ----


def model_comparison():
    # ---- Model comparison ----
    st.subheader("Comparing the four models")
    cv = comparison["CV RMSE (log)"]
    takeaways([
        ("Best in cross-validation", cv.idxmin(), f"RMSE {cv.min():.4f} on the log price"),
        ("Gain over the linear baseline",
         f"{comparison.loc['Linear (Ridge)', 'val Median APE']:.1%} → {final['val Median APE']:.1%}",
         "Median error on the validation set"),
        ("Gap between best and worst model", f"{cv.max() - cv.min():.4f}",
         "The four models are close: the data limits accuracy more than the algorithm"),
    ])
    colors = {m: HIGHLIGHT if m == FINAL_MODEL_NAME else MUTED for m in comparison.index}
    left, right = st.columns(2)
    with left:
        d = comparison["CV RMSE (log)"].sort_values(ascending=False).reset_index()
        fig = px.bar(d, x="CV RMSE (log)", y="model", orientation="h", color="model", color_discrete_map=colors,
                     text=d["CV RMSE (log)"].map("{:.4f}".format), title="Cross-validated error (lower is better)")
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.4f}<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None,
                          bargap=0.45, xaxis_range=[0.25, 0.25 + (d["CV RMSE (log)"].max() - 0.25) * 1.3])
        chart(fig)
        st.caption("RMSE on the log price. The axis starts at 0.25 to make the small differences visible.")
    with right:
        d = (comparison["val Median APE"] * 100).sort_values(ascending=False).reset_index()
        fig = px.bar(d, x="val Median APE", y="model", orientation="h", color="model", color_discrete_map=colors,
                     text=d["val Median APE"].map("{:.1f}%".format), title="Median error on the validation set (%)")
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        style_bars(fig)
        chart(fig)

    # Plain column names and one format per kind of number; "–" where a model was not run on the test set
    table = pd.DataFrame({
        "CV error": comparison["CV RMSE (log)"],
        "Training error": comparison["Train RMSE (log)"],
        "Validation error": comparison["val RMSE (log)"],
        "Validation median error": comparison["val Median APE"],
        "Test error": comparison["test RMSE (log)"],
        "Test median error": comparison["test Median APE"],
        "Test mean abs. error": comparison["test MAE ($)"],
        "Test R²": comparison["test R² (log)"],
        "Tuned settings": comparison["best params"].map(
            lambda s: ", ".join(f"{k} = {v}" for k, v in ast.literal_eval(s).items())),
    })
    table.index.name = "Model"
    rmse_cols = ["CV error", "Training error", "Validation error", "Test error"]
    st.dataframe(table.style.format({**{c: "{:.4f}" for c in rmse_cols},
                                     "Validation median error": "{:.1%}", "Test median error": "{:.1%}",
                                     "Test mean abs. error": "${:,.0f}", "Test R²": "{:.3f}"}, na_rep="–"),
                 width="stretch")
    st.caption("Errors are RMSE on the log price (lower is better). CV = 5-fold cross-validation on the training "
               "set. Test scores come from one evaluation on the 2,800 test sales, after all choices were made.")
    st.markdown(f"""
    - The two ensemble models are best and practically tied; both beat the linear baseline, KNN is weakest.
    - **{FINAL_MODEL_NAME}** was chosen for its best cross-validation score, its smaller gap between training and
      validation error (less overfitting) and its much smaller saved size (1.5 MB vs about 35 MB for the Random Forest).
    - Every model was scored on the test set once, after all choices were made. The picture is the same as on the
      validation set: the two ensemble models tied, ahead of the linear baseline and KNN, and the final model's
      error almost unchanged (0.266 on validation, 0.270 on test).
    """)


def predicted_vs_actual():
    # ---- Predicted vs actual ----
    st.subheader("Predicted vs actual prices on the test set")
    borough = st.selectbox("Show borough", ["All boroughs"] + BOROUGHS)
    d = test_pred if borough == "All boroughs" else test_pred[test_pred["borough_name"] == borough]
    takeaways([
        ("Typical error", f"{d['ape'].median():.1%}", "Half of the predictions are closer than this"),
        ("Within 10% of the real price", f"{(d['ape'] <= 0.10).mean():.0%}", f"of {len(d):,} test sales"),
        ("Within 25% of the real price", f"{(d['ape'] <= 0.25).mean():.0%}", f"of {len(d):,} test sales"),
    ])
    left, right = st.columns(2)
    with left:
        lims = [d[["actual", "predicted"]].min().min() * 0.8, d[["actual", "predicted"]].max().max() * 1.2]
        fig = px.scatter(d, x="actual", y="predicted", log_x=True, log_y=True, opacity=0.5,
                         color_discrete_sequence=[HIGHLIGHT], custom_data=["borough_name", "home_type", "ape"],
                         title=f"Predicted vs actual ({len(d):,} sales)")
        fig.add_shape(type="line", x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1], line=dict(dash="dash", color=MUTED))
        fig.update_traces(marker_size=8, marker_line_width=0, hovertemplate=(
            "%{customdata[0]}, %{customdata[1]}<br>Actual %{x:$,.0f}<br>Predicted %{y:$,.0f}"
            "<br>Error %{customdata[2]:.0%}<extra></extra>"))
        # A few round prices as axis labels; a log axis would otherwise label every step in between
        ticks = [t for t in [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7] if lims[0] <= t <= lims[1]]
        price_axis = dict(tickvals=ticks, ticktext=[money(t) for t in ticks])
        fig.update_layout(xaxis=dict(title="Actual price", **price_axis),
                          yaxis=dict(title="Predicted price", **price_axis))
        chart(fig)
        st.caption("Points on the dashed line are perfect predictions.")
    with right:
        pct = (np.exp(d["residual_log"]) - 1) * 100
        fig = px.histogram(x=pct.clip(-80, 200), nbins=60, color_discrete_sequence=[HIGHLIGHT],
                           title="Prediction error (% of actual price)")
        fig.update_traces(marker_line_width=1, marker_line_color=palette()["card"], hovertemplate="%{y:,} sales<extra></extra>")
        fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
        fig.update_layout(xaxis_title="Predicted / actual − 1 (%)  ·  above 0 = over-predicted", yaxis_title="Sales",
                          bargap=0)
        chart(fig)
        st.caption(f"Median error {d['ape'].median():.1%}. Errors beyond −80% / +200% are shown in the end bins.")


def where_it_struggles():
    # ---- Where the model struggles ----
    def summary(g):
        return pd.Series({"sales": len(g), "Median error": g["ape"].median(),
                          "Bias": np.median(g["predicted"] / g["actual"] - 1)})


    st.subheader("Where the model is accurate and where it struggles")
    bias_notes = st.container()   # filled below, once the errors per borough are computed
    band_by = st.radio("Form the price bands from the", ["actual price", "predicted price"], horizontal=True,
                       help="Grouping by actual price makes cheap sales look over-priced and expensive ones under-priced "
                            "even for a well-calibrated model (regression to the mean). Grouping by predicted price "
                            "is the fair calibration check.")
    by_borough = test_pred.groupby("borough_name").apply(summary, include_groups=False).reindex(BOROUGHS)
    best_b, worst_b = by_borough["Median error"].idxmin(), by_borough["Median error"].idxmax()
    with bias_notes:
        takeaways([
            ("Most accurate borough", best_b, f"Median error {by_borough.loc[best_b, 'Median error']:.1%}"),
            ("Least accurate borough", worst_b,
             f"Median error {by_borough.loc[worst_b, 'Median error']:.1%}, "
             f"only {by_borough.loc[worst_b, 'sales']:.0f} test sales"),
            ("Overall bias", f"{np.median(test_pred['predicted'] / test_pred['actual'] - 1):+.1%}",
             "Median of predicted / actual − 1 over all test sales"),
        ])
    bands = pd.qcut(test_pred["actual" if band_by == "actual price" else "predicted"], 5)
    by_band = test_pred.groupby(bands, observed=True).apply(summary, include_groups=False)
    by_band.index = [f"{money(b.left)} – {money(b.right)}" for b in by_band.index]
    lim = max(by_borough["Bias"].abs().max(), by_band["Bias"].abs().max()) * 1.3

    left, right = st.columns(2)
    for col, data, title in [(left, by_borough, "by borough"), (right, by_band, "by price band")]:
        with col:
            d = data.reset_index(names="group")
            fig = px.bar(d, x="group", y="Bias", text=d["Bias"].map("{:+.0%}".format), custom_data=["sales", "Median error"],
                         color_discrete_sequence=[HIGHLIGHT], title=f"Median bias {title}")
            fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate=(
                "%{x}<br>Bias %{y:+.1%}<br>Median error %{customdata[1]:.1%}<br>%{customdata[0]:,} sales<extra></extra>"))
            fig.add_hline(y=0, line_color=MUTED)
            fig.update_layout(xaxis_title=None, yaxis_title="Median of predicted / actual − 1", yaxis_tickformat="+.0%",
                              yaxis_range=[-lim, lim], bargap=0.45)
            chart(fig)
    st.markdown("""
    - Bars **above zero** mean the model tends to **over-price** that group, bars below zero mean it under-prices.
    - **Grouped by actual price**, the cheapest fifth looks over-priced by about 19% and the most expensive fifth
      under-priced. Most of this is **regression to the mean**: the cheapest actual sales include homes that sold far
      below what similar homes fetch (probably family, distress or poor-condition sales the public record cannot identify).
    - **Grouped by predicted price**, the bias is small in every band (−1% to −4%): when the model predicts a price,
      such homes really sell for close to it. The slight under-pricing comes from the squared-error loss, which
      predicts the mean and is pulled down a little by those unusually cheap sales.
    - **Manhattan is under-priced**: it has very few 1–3 family sales, so the model pulls it towards the average.
    """)


def price_drivers():
    # ---- Feature importance ----
    st.subheader("Which features drive the predictions?")
    d = importance.rename(index=FEATURE_LABELS).sort_values("mean").reset_index()
    location_share = importance.loc[["neighborhood", "zip_code", "borough_name"], "mean"].sum() / importance["mean"].sum()
    takeaways([
        ("Strongest driver", d["feature"].iloc[-1], f"Shuffling it raises the error by {d['mean'].iloc[-1]:.3f}"),
        ("Location in total", f"{location_share:.0%}",
         "Share of the total importance from neighbourhood, ZIP code and borough"),
        ("Least useful feature", d["feature"].iloc[0], "Shuffling it barely changes the error"),
    ])
    fig = px.bar(d, x="mean", y="feature", orientation="h", error_x="std", color_discrete_sequence=[HIGHLIGHT],
                 title="Permutation importance on the test set")
    fig.update_traces(hovertemplate="%{y}: +%{x:.4f} RMSE when shuffled<extra></extra>")
    fig.update_layout(xaxis_title="Increase in test RMSE (log price) when the feature is shuffled", yaxis_title=None,
                      bargap=0.45)
    chart(fig)
    st.caption("Each feature is shuffled 10 times; the bar is the average increase in error and the line its spread. "
               "Location (neighbourhood and ZIP code) dominates, together with building size. The two location columns overlap, so each looks less important alone than location really is.")


# ---- Tabs. Only the open tab is computed, and it is kept in the page address so a link opens the same view ----
TABS = {   # name in the address -> (label, content)
    "comparison": (":material/leaderboard: Model comparison", model_comparison),
    "accuracy": (":material/scatter_plot: Predicted vs actual", predicted_vs_actual),
    "bias": (":material/balance: Where it struggles", where_it_struggles),
    "drivers": (":material/key: What drives the price", price_drivers),
}
# The starting tab comes from the page address only when the page is opened (see the note in dashboard.py)
if "perf_tab" not in st.session_state:
    st.session_state["perf_tab_start"] = TABS[url_value("tab", list(TABS)) or "comparison"][0]
tabs = st.tabs([label for label, _ in TABS.values()], key="perf_tab", on_change="rerun",
               default=st.session_state["perf_tab_start"])
for name, tab, (label, content) in zip(TABS, tabs, TABS.values()):
    if tab.open:
        open_tab = name
        with tab:
            content()

set_url(tab=open_tab if open_tab != "comparison" else None)
