"""Model lab: choose how the data is prepared, pick an algorithm and its settings, train it and see how it scores.

The page uses the same cleaned data, the same train / validation split and the same preprocessing steps as the
notebook (Sections 4b to 5). Models are scored on the validation set; the test set is not used here.
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from threadpoolctl import threadpool_limits

from utils import (FINAL_MODEL_NAME, HIGHLIGHT, MUTED, card, chart, load_homes, load_results, money, page_footer,
                   page_header, takeaways)

RANDOM_STATE = 42   # the same seed as the notebook
# Training uses at most two threads. With more, the tree models get slower, not faster, on a small server:
# the threads compete for the same processor (measured: minutes instead of seconds on the hosting server).
TRAIN_THREADS = 2
NUMERIC = ["log_gross_sqft", "log_land_sqft", "floor_area_ratio", "building_age"]
ONEHOT = ["borough_name", "building_class_at_time_of_sale", "sale_quarter"]
LOCATION_OPTIONS = {   # label -> columns that are target-encoded
    "Borough only": [],
    "Borough + neighbourhood": ["neighborhood"],
    "Borough + neighbourhood + ZIP code": ["neighborhood", "zip_code"],
}
# The settings found by the grid search in the notebook (Section 5)
TUNED = {
    "Linear Regression (Ridge)": {"alpha": 0.1},
    "K-Nearest Neighbours": {"n_neighbors": 31, "weights": "distance"},
    "Random Forest": {"n_estimators": 300, "max_features": 0.25, "min_samples_leaf": 3},
    "Gradient Boosting": {"learning_rate": 0.01, "max_iter": 1000, "max_leaf_nodes": 31, "min_samples_leaf": 20},
}
ABOUT = {
    "Linear Regression (Ridge)": "A straight-line model: each feature adds or subtracts a fixed amount. Fast and easy "
                                 "to explain, and the baseline every other model has to beat.",
    "K-Nearest Neighbours": "Prices a home as the average of the most similar homes sold, the way valuers use "
                            "comparables. It needs scaled features, because it measures distances.",
    "Random Forest": "Averages many deep decision trees, each trained on a random sample of the homes. It handles "
                     "non-linear effects well and is hard to break.",
    "Gradient Boosting": "Builds small trees one after another, each correcting the errors of the ones before. "
                         "This is the family of the final model.",
}


@st.cache_data(show_spinner="Preparing the modelling data…")
def modelling_data():
    """Features engineered as in notebook Section 4b, and the training and validation rows of Section 4c.

    The split is rebuilt with the notebook's code and seed rather than read from the saved file, so the rows come
    in the same order as in the notebook and a model trained here gets the same scores as there."""
    homes = load_homes()
    X = pd.DataFrame({
        "log_gross_sqft": np.log(homes["gross_square_feet"]),
        "log_land_sqft": np.log(homes["land_square_feet"]),
        "floor_area_ratio": homes["floor_area_ratio"],
        "building_age": homes["building_age"],
        "has_commercial": homes["has_commercial"],
        "home_type": homes["home_type"],
        "borough_name": homes["borough_name"],
        "building_class_at_time_of_sale": homes["building_class_at_time_of_sale"],
        "sale_quarter": homes["sale_quarter"],
        "neighborhood": homes["neighborhood"],
        "zip_code": homes["zip_code"],
    })
    price = homes["sale_price"]
    rows = np.arange(len(homes))
    band = pd.qcut(np.log(price), q=10, labels=False)   # 10 price bands, used only to stratify the split
    train, rest, _, band_rest = train_test_split(rows, band, test_size=0.30, stratify=band, random_state=RANDOM_STATE)
    val, _ = train_test_split(rest, test_size=0.50, stratify=band_rest, random_state=RANDOM_STATE)
    return X, price, train, val


def make_preprocessor(scale: bool, location: list) -> ColumnTransformer:
    """The preprocessing of notebook Section 4d, with two switches: scaling on or off, and the location columns."""
    def steps(*first):
        return Pipeline([*first, ("scale", StandardScaler())]) if scale else Pipeline([*first])

    parts = [
        ("num", steps(("impute", SimpleImputer(strategy="median"))), NUMERIC),
        ("size_missing", MissingIndicator(), ["log_gross_sqft"]),
        ("bin", SimpleImputer(strategy="most_frequent"), ["has_commercial"]),
        ("ord", steps(("encode", OrdinalEncoder(categories=[["1-family", "2-family", "3-family"]]))), ["home_type"]),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False), ONEHOT),
    ]
    if location:
        encoder = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        parts.append(("te", steps(("encode", encoder)), location))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def make_model(algorithm: str, params: dict):
    if algorithm == "Linear Regression (Ridge)":
        return Ridge(**params)
    if algorithm == "K-Nearest Neighbours":
        return KNeighborsRegressor(**params)
    if algorithm == "Random Forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=TRAIN_THREADS, **params)
    return HistGradientBoostingRegressor(early_stopping=False, random_state=RANDOM_STATE, **params)


@st.cache_data(show_spinner=False, max_entries=30)
def train_and_score(algorithm: str, params: tuple, log_target: bool, scale: bool, location_label: str) -> dict:
    """Train on the training set and score on the validation set. Cached, so a repeated run is instant."""
    X, price, train, val = modelling_data()
    pipe = Pipeline([("prep", make_preprocessor(scale, LOCATION_OPTIONS[location_label])),
                     ("model", make_model(algorithm, dict(params)))])
    start = time.perf_counter()
    with threadpool_limits(limits=TRAIN_THREADS):
        pipe.fit(X.iloc[train], np.log(price.iloc[train]) if log_target else price.iloc[train])
    seconds = time.perf_counter() - start

    def predict_dollars(rows):
        with threadpool_limits(limits=TRAIN_THREADS):
            pred = pipe.predict(X.iloc[rows])
        # a model trained on dollars can predict a price near or below zero; keep it positive for the log scale
        return np.exp(pred) if log_target else np.clip(pred, 10_000, None)

    def scores(rows):
        actual, pred = price.iloc[rows].to_numpy(), predict_dollars(rows)
        ape = np.abs(pred - actual) / actual
        return {"rmse_log": float(np.sqrt(mean_squared_error(np.log(actual), np.log(pred)))),
                "r2_log": float(r2_score(np.log(actual), np.log(pred))),
                "median_ape": float(np.median(ape)), "mae": float(mean_absolute_error(actual, pred))}

    return {"train": scores(train), "val": scores(val), "seconds": seconds,
            "columns": int(pipe.named_steps["prep"].transform(X.iloc[train[:50]]).shape[1]),
            "actual": price.iloc[val].to_numpy(), "predicted": predict_dollars(val),
            "borough": X["borough_name"].iloc[val].to_numpy()}


# ---------------------------------------------------------------------------------------------------------------
page_header("Model lab", "Choose how the data is prepared, pick an algorithm and its settings, then train it and see "
                         "how it scores")
_, _, train_rows, val_rows = modelling_data()
comparison = load_results("model_comparison")
final = comparison.loc[FINAL_MODEL_NAME]
n_train, n_val = len(train_rows), len(val_rows)

left, right = st.columns(2, gap="medium")
with left, card("lab_prep", height="stretch"):
    st.markdown("#### 1. Data preparation")
    log_target = st.toggle("Predict the log of the price", value=True,
                           help="Prices are strongly right-skewed. On the log scale a 10% error counts the same for a "
                                "cheap and an expensive home. Switch it off to train on dollars and see the difference.")
    scale = st.toggle("Scale the numeric features", value=True,
                      help="Puts every numeric column on the same scale (mean 0, standard deviation 1). It matters "
                           "for K-Nearest Neighbours and the linear model, and not at all for tree models.")
    location_label = st.radio("Location detail given to the model", list(LOCATION_OPTIONS), index=2,
                              help="Neighbourhood and ZIP code are target-encoded: each is replaced by the typical "
                                   "price of its sales, learned from the training data only.")
    st.caption("Always applied, as in the notebook: sizes log-transformed, missing sizes filled with the training "
               "median and flagged, home type as an ordered number, borough, building class and quarter one-hot "
               "encoded.")

with right, card("lab_model", height="stretch"):
    st.markdown("#### 2. Algorithm and settings")
    algorithm = st.selectbox("Algorithm", list(TUNED), index=3)
    st.caption(ABOUT[algorithm])
    tuned = TUNED[algorithm]
    use_tuned = st.toggle("Use the settings found by the grid search in the notebook", value=True, key=f"tuned_{algorithm}")
    c1, c2 = st.columns(2)
    if algorithm == "Linear Regression (Ridge)":
        alphas = [0.01, 0.1, 1, 3, 10, 30, 100, 1000]
        params = {"alpha": c1.select_slider("alpha (strength of the penalty)", alphas, value=tuned["alpha"],
                                            disabled=use_tuned,
                                            help="Larger values shrink the coefficients more: a simpler model.")}
    elif algorithm == "K-Nearest Neighbours":
        params = {"n_neighbors": c1.slider("Number of neighbours (k)", 1, 60, tuned["n_neighbors"], disabled=use_tuned,
                                           help="Small k follows single sales (overfits); large k averages too many "
                                                "homes (underfits)."),
                  "weights": c2.radio("Weights", ["uniform", "distance"], index=1, disabled=use_tuned,
                                      help="distance: closer neighbours count more.")}
    elif algorithm == "Random Forest":
        params = {"n_estimators": c1.slider("Number of trees", 20, 300, tuned["n_estimators"], step=20,
                                            disabled=use_tuned, help="More trees give a steadier average, but train "
                                                                     "more slowly."),
                  "min_samples_leaf": c2.slider("Smallest leaf (homes)", 1, 30, tuned["min_samples_leaf"],
                                                disabled=use_tuned, help="Larger leaves stop the trees from fitting "
                                                                         "single unusual sales."),
                  "max_features": c1.select_slider("Share of columns tried per split", [0.15, 0.25, 0.33, 0.5, 1.0],
                                                   value=tuned["max_features"], disabled=use_tuned,
                                                   help="Fewer columns per split make the trees less alike.")}
    else:
        params = {"learning_rate": c1.select_slider("Learning rate", [0.01, 0.03, 0.1, 0.3], value=tuned["learning_rate"],
                                                    disabled=use_tuned, help="How much each new tree corrects the "
                                                                             "previous ones."),
                  "max_iter": c2.slider("Number of trees", 50, 2000, tuned["max_iter"], step=50, disabled=use_tuned,
                                        help="With a small learning rate the model needs more trees."),
                  "max_leaf_nodes": c1.select_slider("Leaves per tree", [7, 15, 31, 63], value=tuned["max_leaf_nodes"],
                                                     disabled=use_tuned, help="Bigger trees learn more complex "
                                                                              "interactions."),
                  "min_samples_leaf": c2.slider("Smallest leaf (homes)", 5, 100, tuned["min_samples_leaf"], step=5,
                                                disabled=use_tuned)}
    if use_tuned:
        params = dict(tuned)

st.caption(f"The model is trained on the {n_train:,} training sales and scored on the {n_val:,} validation sales "
           "it has not seen. The test set is not used on this page: it is kept for the final evaluation on the "
           "Model performance page.")

run = st.button("Train the model", type="primary", icon=":material/play_arrow:")
settings = (algorithm, tuple(sorted(params.items())), log_target, scale, location_label)
if run:
    st.session_state["lab_settings"] = settings
if "lab_settings" not in st.session_state:
    st.info("Choose the preparation and the algorithm above, then press **Train the model**.")
    page_footer()
    st.stop()

# ---- Result of the last trained model (kept on screen while the choices above are changed) ----
shown = st.session_state["lab_settings"]
with st.spinner(f"Training {shown[0]}… A forest or a boosting model with many trees can take up to a minute."):
    result = train_and_score(*shown)
if shown != settings:
    st.warning("The choices above have changed since this model was trained. Press **Train the model** to update "
               "the result.")

label = f"{shown[0]} · " + ", ".join(f"{k} = {v}" for k, v in shown[1])
val, trn = result["val"], result["train"]
runs = st.session_state.setdefault("lab_runs", {})
runs[shown] = {"Algorithm": shown[0], "Settings": ", ".join(f"{k} = {v}" for k, v in shown[1]),
               "Log price": "yes" if shown[2] else "no", "Scaled": "yes" if shown[3] else "no", "Location": shown[4],
               "Validation RMSE (log)": val["rmse_log"], "Median error": val["median_ape"],
               "Training RMSE (log)": trn["rmse_log"], "Seconds": result["seconds"]}

st.subheader("Result on the validation set")
st.caption(f"{label} · {result['columns']} input columns after preparation · trained in {result['seconds']:.1f} s")
gap = trn["rmse_log"] - val["rmse_log"]


def against_final(value, final_value, fmt):
    """Words, not arrows: for an error, lower is better, so an up arrow would read the wrong way."""
    if abs(value - final_value) < 0.0005:
        return f"the same as the final model ({final_value:{fmt}})"
    return f"{'better' if value < final_value else 'worse'} than the final model ({final_value:{fmt}})"


takeaways([
    ("Typical error", f"{val['median_ape']:.1%}", against_final(val["median_ape"], final["val Median APE"], ".1%")),
    ("RMSE on the log price", f"{val['rmse_log']:.4f}",
     against_final(val["rmse_log"], final["val RMSE (log)"], ".4f")),
    ("Mean absolute error", money(val["mae"]), f"R² on the log price {val['r2_log']:.3f}"),
    ("Training vs validation", f"{trn['rmse_log']:.4f} vs {val['rmse_log']:.4f}",
     "a much lower training error means the model has memorised the training sales" if gap < -0.03
     else "the two are close: little overfitting"),
])
st.caption("The final model is the tuned Gradient Boosting model of the notebook, scored on the same validation "
           "sales. For both errors, lower is better.")

left, right = st.columns(2)
with left:
    d = pd.DataFrame({"actual": result["actual"], "predicted": result["predicted"], "borough": result["borough"]})
    lims = [d[["actual", "predicted"]].min().min() * 0.8, d[["actual", "predicted"]].max().max() * 1.2]
    fig = px.scatter(d, x="actual", y="predicted", log_x=True, log_y=True, opacity=0.5, custom_data=["borough"],
                     color_discrete_sequence=[HIGHLIGHT], title=f"Predicted vs actual ({len(d):,} validation sales)")
    fig.add_shape(type="line", x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1], line=dict(dash="dash", color=MUTED))
    fig.update_traces(marker_size=7, marker_line_width=0, hovertemplate=(
        "%{customdata[0]}<br>Actual %{x:$,.0f}<br>Predicted %{y:$,.0f}<extra></extra>"))
    # A few round prices as axis labels; a log axis would otherwise label every step in between
    ticks = [t for t in [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7] if lims[0] <= t <= lims[1]]
    price_axis = dict(tickvals=ticks, ticktext=[money(t) for t in ticks])
    fig.update_layout(xaxis=dict(title="Actual price", **price_axis),
                      yaxis=dict(title="Predicted price", **price_axis))
    chart(fig)
    st.caption("Points on the dashed line are perfect predictions.")
with right:
    error = (d["predicted"] / d["actual"] - 1) * 100
    fig = px.histogram(x=error.clip(-80, 200), nbins=60, color_discrete_sequence=[HIGHLIGHT],
                       title="Prediction error (% of actual price)")
    fig.update_traces(hovertemplate="%{y:,} sales<extra></extra>")
    fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
    fig.update_layout(xaxis_title="Predicted / actual − 1 (%)  ·  above 0 = over-predicted", yaxis_title="Sales",
                      bargap=0.05)
    chart(fig)
    st.caption("A narrow peak around 0 is what a good model looks like. Errors beyond −80% / +200% are shown in the "
               "end bins.")

# ---- Every model trained in this visit, best first ----
st.subheader("Models you have trained")
history = pd.DataFrame(runs.values()).sort_values("Validation RMSE (log)").reset_index(drop=True)
st.dataframe(history.style.format({"Validation RMSE (log)": "{:.4f}", "Training RMSE (log)": "{:.4f}",
                                   "Median error": "{:.1%}", "Seconds": "{:.1f}"}),
             hide_index=True, width="stretch")
left, right = st.columns([3, 1], vertical_alignment="center")
left.caption("Sorted by validation error, best first. Change one thing at a time and train again to see what each "
             "choice is worth. The list is kept until you leave or reload the app.")
if right.button("Clear the list", icon=":material/delete:", width="stretch"):
    st.session_state.pop("lab_runs", None)
    st.rerun()
