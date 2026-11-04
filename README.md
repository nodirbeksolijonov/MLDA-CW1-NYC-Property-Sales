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

## Project structure

```
ML_CW/
├── data/
│   ├── 2025_*.xlsx                     # raw borough sales files
│   └── processed/homes_2025_clean.csv  # cleaned dataset exported by the notebook (used by the app)
├── notebooks/
│   └── MLDA_CW1_NYC_Property_Sales.ipynb   # main analysis (EDA → preparation → models → evaluation)
├── models/
│   ├── final_model.joblib              # final pipeline (preprocessing + Gradient Boosting), 0.5 MB
│   └── results/                        # model comparison, test predictions, feature importance
├── app/
│   ├── streamlit_app.py                # app entry point (multi-page navigation)
│   ├── utils.py                        # shared loading, feature engineering and styling
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

`requirements.txt` lists the direct dependencies; `requirements-lock.txt` pins every package, including
sub-dependencies, to the versions the results were produced with. Running the notebook in a fresh environment
built from the lock file reproduces all reported results exactly on macOS. Inside the Linux Docker container the
final Gradient Boosting model, its hyperparameters and all of its results are identical, and so are the Ridge and
KNN scores (to 6 decimals). Only the Random Forest reference scores differ, in the 4th decimal (normal
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

The app opens at http://localhost:8501 and has five pages:

| Page | What it shows |
|---|---|
| **Overview** | Project summary and headline results |
| **Market dashboard** | Filters by borough, home type and quarter; prices by borough, price distribution, monthly trends, most/least expensive neighbourhoods, size vs price |
| **Price predictor** | Estimated 2025 price for a home you describe, with a realistic range and comparable sales |
| **Model performance** | Comparison of the four tuned models, test-set accuracy, errors by borough and price band, feature importance |
| **Data & method** | Data source, cleaning steps, features and limitations |

The app reads the files saved by the notebook (`data/processed/`, `models/`). They are included in the repository, so the app works without re-running the notebook. Re-running the notebook regenerates them.

## Licence

Code is released under the MIT Licence — see [LICENSE](LICENSE).
The dataset remains subject to the NYC Open Data Terms of Use.
