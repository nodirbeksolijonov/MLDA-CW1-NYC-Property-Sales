# Predicting Residential Property Sale Prices in New York City (2025)

Coursework 1 for **6COSC017C Machine Learning and Data Analytics**.

An end-to-end machine learning project that predicts the sale price of 1–3 family
homes in New York City from their physical characteristics, location and sale timing,
and identifies which factors drive price the most.

## Dataset

**NYC Department of Finance — Annualized Property Sales, calendar year 2025**
(one Excel file per borough: Manhattan, Bronx, Brooklyn, Queens, Staten Island).

- Source: https://www.nyc.gov/site/finance/property/property-annualized-sales-update.page
- Licence / terms: NYC Open Data Terms of Use (free public use with attribution).
- The data was downloaded directly from the NYC Department of Finance website (not from Kaggle).

## Project structure

```
ML_CW/
├── data/                               # raw borough sales files (2025_*.xlsx)
├── notebooks/
│   └── MLDA_CW1_NYC_Property_Sales.ipynb   # main analysis (EDA → preparation → models → evaluation)
├── models/                             # trained models saved by the notebook (not committed)
├── requirements.txt                    # pinned dependencies
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

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter notebook notebooks/MLDA_CW1_NYC_Property_Sales.ipynb
```

Run all cells from top to bottom. The notebook reads the raw files from `../data/`, so keep the folder structure as it is. A fixed random seed (`RANDOM_STATE = 42`) makes results reproducible.

## Running the Streamlit app

*(Coming soon.)*

## Licence

Code is released under the MIT Licence — see [LICENSE](LICENSE).
The dataset remains subject to the NYC Open Data Terms of Use.
