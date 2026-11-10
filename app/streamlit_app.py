"""Entry point: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

# make app/utils.py importable from the pages, wherever the app is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="NYC Home Prices 2025", page_icon="🏠", layout="wide")

# Area unit switch, shared by every page through st.session_state["area_unit"]
st.sidebar.radio("Area unit", ["ft²", "m²"], key="area_unit", horizontal=True,
                 help="The source data records sizes in square feet; m² values are converted (1 m² = 10.764 ft²)")

pages = st.navigation([
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/dashboard.py", title="Market dashboard", icon="📊"),
    st.Page("views/predictor.py", title="Price predictor", icon="💲"),
    st.Page("views/performance.py", title="Model performance", icon="🎯"),
    st.Page("views/data.py", title="Data & method", icon="📄"),
])
pages.run()
