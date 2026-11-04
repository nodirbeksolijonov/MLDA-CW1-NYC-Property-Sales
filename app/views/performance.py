import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (BOROUGHS, FEATURE_LABELS, FINAL_MODEL_NAME, HIGHLIGHT, MUTED, load_results, money,
                   page_header, style_bars)

comparison = load_results("model_comparison")
test_pred = load_results("test_predictions")
importance = load_results("permutation_importance")
final = comparison.loc[FINAL_MODEL_NAME]
PLOT = dict(theme="streamlit", width="stretch")

page_header("Model performance",
            f"Four tuned models compared, and the final {FINAL_MODEL_NAME} model evaluated on 2,800 unseen test sales")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Median error (test)", f"{final['test Median APE']:.1%}")
k2.metric("Mean absolute error (test)", money(final["test MAE ($)"]))
k3.metric("R² on log price (test)", f"{final['test R² (log)']:.3f}")
k4.metric("RMSE on log price (test)", f"{final['test RMSE (log)']:.3f}")

# ---- Model comparison ----
st.subheader("1. Comparing the four models")
colors = {m: HIGHLIGHT if m == FINAL_MODEL_NAME else MUTED for m in comparison.index}
left, right = st.columns(2)
with left:
    d = comparison["CV RMSE (log)"].sort_values(ascending=False).reset_index()
    fig = px.bar(d, x="CV RMSE (log)", y="model", orientation="h", color="model", color_discrete_map=colors,
                 text=d["CV RMSE (log)"].map("{:.4f}".format), title="Cross-validated RMSE (log price, lower is better)")
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None,
                      bargap=0.45, xaxis_range=[0.25, d["CV RMSE (log)"].max() * 1.04])
    st.plotly_chart(fig, **PLOT)
    st.caption("The axis starts at 0.25 to make the small differences visible.")
with right:
    d = (comparison["val Median APE"] * 100).sort_values(ascending=False).reset_index()
    fig = px.bar(d, x="val Median APE", y="model", orientation="h", color="model", color_discrete_map=colors,
                 text=d["val Median APE"].map("{:.1f}%".format), title="Median error on the validation set (%)")
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    style_bars(fig)
    st.plotly_chart(fig, **PLOT)

table = comparison[["best params", "CV RMSE (log)", "Train RMSE (log)", "val RMSE (log)", "val Median APE",
                    "test RMSE (log)", "test Median APE", "test MAE ($)", "test R² (log)"]]
st.dataframe(table, width="stretch", column_config={
    "CV RMSE (log)": st.column_config.NumberColumn(format="%.4f"),
    "Train RMSE (log)": st.column_config.NumberColumn(format="%.4f"),
    "val RMSE (log)": st.column_config.NumberColumn(format="%.4f"),
    "test RMSE (log)": st.column_config.NumberColumn(format="%.4f"),
    "val Median APE": st.column_config.NumberColumn(format="percent"),
    "test Median APE": st.column_config.NumberColumn(format="percent"),
    "test MAE ($)": st.column_config.NumberColumn(format="dollar"),
    "test R² (log)": st.column_config.NumberColumn(format="%.3f"),
})
st.markdown(f"""
- The two ensemble models are best and practically tied; both beat the linear baseline, KNN is weakest.
- **{FINAL_MODEL_NAME}** was chosen for its best cross-validation score, its smaller gap between training and
  validation error (less overfitting) and its much smaller saved size (0.5 MB vs about 35 MB for the Random Forest).
- Test results were measured only for the final model and two reference models, once, after all choices were made.
""")

# ---- Predicted vs actual ----
st.subheader("2. Predicted vs actual prices on the test set")
borough = st.selectbox("Show borough", ["All boroughs"] + BOROUGHS)
d = test_pred if borough == "All boroughs" else test_pred[test_pred["borough_name"] == borough]
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
    fig.update_layout(xaxis_title="Actual price", yaxis_title="Predicted price",
                      xaxis_tickformat="$,.2s", yaxis_tickformat="$,.2s")
    st.plotly_chart(fig, **PLOT)
    st.caption("Points on the dashed line are perfect predictions.")
with right:
    pct = (np.exp(d["residual_log"]) - 1) * 100
    fig = px.histogram(x=pct.clip(-80, 200), nbins=60, color_discrete_sequence=[HIGHLIGHT],
                       title="Prediction error (% of actual price)")
    fig.update_traces(marker_line_width=1, marker_line_color="white", hovertemplate="%{y:,} sales<extra></extra>")
    fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
    fig.update_layout(xaxis_title="Predicted / actual − 1 (%)  ·  above 0 = over-predicted", yaxis_title="Sales",
                      bargap=0)
    st.plotly_chart(fig, **PLOT)
    st.caption(f"Median error {d['ape'].median():.1%}. Errors beyond −80% / +200% are shown in the end bins.")


# ---- Where the model struggles ----
def summary(g):
    return pd.Series({"sales": len(g), "Median error": g["ape"].median(),
                      "Bias": np.median(g["predicted"] / g["actual"] - 1)})


st.subheader("3. Where the model is accurate and where it struggles")
by_borough = test_pred.groupby("borough_name").apply(summary, include_groups=False).reindex(BOROUGHS)
bands = pd.qcut(test_pred["actual"], 5)
by_band = test_pred.groupby(bands, observed=True).apply(summary, include_groups=False)
by_band.index = [f"{money(b.left)} – {money(b.right)}" for b in by_band.index]

left, right = st.columns(2)
for col, data, title in [(left, by_borough, "By borough"), (right, by_band, "By price band (fifths of actual price)")]:
    with col:
        d = data.reset_index(names="group")
        fig = px.bar(d, x="group", y="Bias", text=d["Bias"].map("{:+.0%}".format), custom_data=["sales", "Median error"],
                     color_discrete_sequence=[HIGHLIGHT], title=f"{title}: median bias")
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate=(
            "%{x}<br>Bias %{y:+.1%}<br>Median error %{customdata[1]:.1%}<br>%{customdata[0]:,} sales<extra></extra>"))
        fig.add_hline(y=0, line_color=MUTED)
        lim = d["Bias"].abs().max() * 1.3
        fig.update_layout(xaxis_title=None, yaxis_title="Median of predicted / actual − 1", yaxis_tickformat="+.0%",
                          yaxis_range=[-lim, lim], bargap=0.45)
        st.plotly_chart(fig, **PLOT)
st.markdown("""
- Bars **above zero** mean the model tends to **over-price** that group, bars below zero mean it under-prices.
- The **cheapest fifth** of sales is over-priced by about 20%: many are probably below-market sales (family,
  distress, poor condition) that the public record cannot identify.
- **Expensive homes and Manhattan are under-priced**: they are rare in the data, so the model pulls them
  towards the average.
""")

# ---- Feature importance ----
st.subheader("4. Which features drive the predictions?")
d = importance.rename(index=FEATURE_LABELS).sort_values("mean").reset_index()
fig = px.bar(d, x="mean", y="feature", orientation="h", error_x="std", color_discrete_sequence=[HIGHLIGHT],
             title="Permutation importance on the test set")
fig.update_traces(hovertemplate="%{y}: +%{x:.4f} RMSE when shuffled<extra></extra>")
fig.update_layout(xaxis_title="Increase in test RMSE (log price) when the feature is shuffled", yaxis_title=None,
                  bargap=0.45)
st.plotly_chart(fig, **PLOT)
st.caption("Each feature is shuffled 10 times; the bar is the average increase in error and the line its spread. "
           "Location (neighbourhood and ZIP code) dominates, together with building size. The two location columns overlap, so each looks less important alone than location really is.")
