"""Entry point: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

# make app/utils.py importable from the pages, wherever the app is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ASSETS, FINAL_MODEL_NAME, apply_style  # noqa: E402

st.set_page_config(page_title="NYC Home Prices 2025", page_icon=str(ASSETS / "icon.svg"), layout="wide")
apply_style()
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

pages = st.navigation({
    "": [st.Page("views/home.py", title="Overview", icon=":material/home:", default=True)],
    "Explore": [
        st.Page("views/dashboard.py", title="Market dashboard", icon=":material/monitoring:"),
        st.Page("views/predictor.py", title="Price predictor", icon=":material/calculate:"),
    ],
    "Model": [
        st.Page("views/performance.py", title="Model performance", icon=":material/query_stats:"),
        st.Page("views/data.py", title="Data & method", icon=":material/description:"),
    ],
})

# Area unit switch, shared by every page through st.session_state["area_unit"]
st.sidebar.radio("Area unit", ["ft²", "m²"], key="area_unit", horizontal=True,
                 help="The source data records sizes in square feet; m² values are converted (1 m² = 10.764 ft²)")
st.sidebar.html(f'''<div class="sidebar-note"><b>Data</b><br>NYC Department of Finance, annualized sales 2025
<br><br><b>Model</b><br>{FINAL_MODEL_NAME}, log sale price</div>''')

pages.run()
