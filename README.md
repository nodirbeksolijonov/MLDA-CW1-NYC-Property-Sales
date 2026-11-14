# Predicting Residential Property Sale Prices in New York City (2025)

Coursework 1 for **6COSC017C Machine Learning and Data Analytics**.

An end-to-end machine learning project that predicts the sale price of 1–3 family
homes in New York City from their physical characteristics, location and sale timing,
and identifies which factors drive price the most.

**Live app:** https://nycprices.streamlit.app/

## Dataset

**NYC Department of Finance — Annualized Property Sales, calendar year 2025**
(one Excel file per borough: Manhattan, Bronx, Brooklyn, Queens, Staten Island).

- Source: https://www.nyc.gov/site/finance/property/property-annualized-sales-update.page
- Licence / terms: NYC Open Data Terms of Use (free public use with attribution).
- The data was downloaded directly from the NYC Department of Finance website (not from Kaggle).

**ZIP code boundaries** (used only for the maps): NYC Open Data, *Modified Zip Code Tabulation Areas (MODZCTA)*, dataset `pri4-ifjk`,
https://data.cityofnewyork.us/Health/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk.
Shapes simplified to about 50 m and saved as `data/nyc_zip_boundaries.geojson`; each area lists the ZIP codes it covers.

## Project structure

```
ML_CW/
├── data/
│   ├── 2025_*.xlsx                     # raw borough sales files
│   ├── nyc_zip_boundaries.geojson      # ZIP code boundary areas for the maps (NYC Open Data)
│   └── processed/homes_2025_clean.csv  # cleaned dataset exported by the notebook (used by the app)
├── notebooks/
│   └── MLDA_CW1_NYC_Property_Sales.ipynb   # main analysis (EDA → preparation → models → evaluation)
├── models/
│   ├── final_model.joblib              # final pipeline (preprocessing + Gradient Boosting), 1.5 MB
│   └── results/                        # model comparison, test predictions, feature importance
├── app/
│   ├── streamlit_app.py                # app entry point (multi-page navigation)
│   ├── utils.py                        # shared loading, feature engineering and styling
│   ├── assets/                         # stylesheet, logo and icon
│   └── views/                          # overview, dashboard, predictor, model performance, data & method
├── .streamlit/config.toml              # app theme
├── requirements.txt                    # pinned direct dependencies
├── requirements-lock.txt               # full lock file (all sub-dependencies, exact versions)
├── Dockerfile                          # containerised environment
├── LICENSE
└── README.md
```

## Setup

Requires **Python 3.13**.

```bash
# 1. Clone the repository
git clone https://github.com/nodirbeksolijonov/MLDA-CW1-NYC-Property-Sales.git
cd MLDA-CW1-NYC-Property-Sales

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the exact package versions
pip install -r requirements-lock.txt
```

XGBoost and LightGBM (notebook Section 5.7) need the OpenMP runtime. It is included on Windows and in the
Docker image; on macOS install it once with `brew install libomp`, and on Debian/Ubuntu with
`sudo apt-get install libgomp1`. The Streamlit app does not use these two libraries.

`requirements.txt` lists the direct dependencies; `requirements-lock.txt` pins every package, including
sub-dependencies, to the versions the results were produced with. Running the notebook in a fresh environment
built from the lock file reproduces all reported results exactly on macOS. Inside the Linux Docker container the
final Gradient Boosting model, its hyperparameters and all of its results are identical, and so are the Ridge and
KNN scores (to 6 decimals). Only the Random Forest and XGBoost scores differ, in the 4th decimal (normal
floating-point differences between operating systems), which does not change any ranking or conclusion.

### Alternative: Docker

```bash
docker build -t nyc-home-prices .

# Run the Streamlit app at http://localhost:8501
docker run --rm -p 8501:8501 nyc-home-prices

# Re-run the whole notebook inside the container (results are written back to your folder)
docker run --rm -v "$(pwd)":/project nyc-home-prices \
    jupyter nbconvert --to notebook --execute --inplace notebooks/MLDA_CW1_NYC_Property_Sales.ipynb
```

## Running the notebook

```bash
jupyter notebook notebooks/MLDA_CW1_NYC_Property_Sales.ipynb
```

Run all cells from top to bottom. The notebook reads the raw files from `../data/`, so keep the folder structure as it is. A fixed random seed (`RANDOM_STATE = 42`) makes results reproducible.

## Running the Streamlit app

The deployed version is at https://nycprices.streamlit.app/. To run it locally:

```bash
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501 and has six pages:

| Page | What it shows |
|---|---|
| **Overview** | Landing page with a price map of the city, headline results and a summary of how the model was built |
| **Market dashboard** | Filters by borough, home type and quarter, then five tabs: prices (by borough, distribution, most/least expensive neighbourhoods, size vs price), trends (by month and by decade built), ZIP code price map (actual vs learned by the model), feature correlations, and the filtered data; ft²/m² switch in the sidebar |
| **Price predictor** | Estimated 2025 price for a home you describe, with a realistic range, what-if cards (the same home with one input changed) and comparable sales |
| **Model performance** | Four tabs: comparison of the four tuned models, predicted vs actual on the test set, errors by borough and price band, feature importance |
| **Model lab** | Choose the data preparation (log target, scaling, location detail), pick one of the four algorithms and its hyperparameters, train it on the training set and see its validation scores, charts and a list of the models trained so far |
| **Data & method** | Data source, cleaning steps, features and limitations |

Other features of the app:

- **Shareable links.** The page address follows the dashboard filters, the open tab and the predictor inputs, so a copied link opens the same view (for example `…/dashboard?borough=Queens&tab=map`).
- **Downloads.** The filtered sales and the comparable sales can be downloaded as CSV.
- **Light and dark theme.** The app starts in the visitor's system setting and has a Light / Dark switch in the sidebar; charts and the map are restyled to match.
- **Speed.** Only the open tab is computed.

The app reads the files saved by the notebook (`data/processed/`, `models/`). They are included in the repository, so the app works without re-running the notebook. Re-running the notebook regenerates them.

## Licence

Code is released under the MIT Licence — see [LICENSE](LICENSE).
The dataset remains subject to the NYC Open Data Terms of Use.
