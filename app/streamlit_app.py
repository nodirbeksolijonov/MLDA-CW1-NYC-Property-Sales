"""Entry point: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

# make app/utils.py importable from the pages, wherever the app is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import ASSETS, DARK, FINAL_MODEL_NAME, apply_style, page_footer, palette  # noqa: E402

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
        st.Page("views/lab.py", title="Model lab", icon=":material/science:"),
        st.Page("views/data.py", title="Data & method", icon=":material/description:"),
    ],
})

# Browser tab: the page name first, e.g. "Price predictor · NYC Home Prices"
st.set_page_config(page_title="NYC Home Prices 2025" if pages.title == "Overview" else f"{pages.title} · NYC Home Prices",
                   page_icon=str(ASSETS / "icon.svg"), layout="wide")

# Area unit switch, shared by every page through st.session_state["area_unit"]; a shared link can ask for m²
if "area_unit" not in st.session_state and st.query_params.get("unit") == "m2":
    st.session_state["area_unit"] = "m²"
st.sidebar.radio("Area unit", ["ft²", "m²"], key="area_unit", horizontal=True,
                 help="The source data records sizes in square feet; m² values are converted (1 m² = 10.764 ft²)")

# Light / dark switch. It always shows the theme in use (the visitor's system setting until they choose one)
THEME_LABELS = {"Light": ":material/light_mode: Light", "Dark": ":material/dark_mode: Dark"}
theme_in_use = "Dark" if palette() is DARK else "Light"
if "theme_request" not in st.session_state:
    st.session_state["theme_choice"] = theme_in_use
st.sidebar.segmented_control(
    "Theme", list(THEME_LABELS), key="theme_choice", format_func=THEME_LABELS.get, width="stretch",
    on_change=lambda: st.session_state.update(theme_request=st.session_state["theme_choice"]))
theme_request = st.session_state.pop("theme_request", None)

st.sidebar.html(f'''<div class="sidebar-note"><b>Data</b><br>NYC Department of Finance, annualized sales 2025
<br><br><b>Model</b><br>{FINAL_MODEL_NAME}, log sale price</div>''')

pages.run()
page_footer()

if theme_request in THEME_LABELS and theme_request != theme_in_use:
    # Streamlit keeps the visitor's theme choice in the browser's local storage, one entry per page address, and
    # reads it when the page loads. Store the choice for every page, then reload; the page address (which carries
    # the filters and inputs) brings the same view back in the new theme.
    st.html(f"""<script>
(() => {{
  const pages = ["", "dashboard", "predictor", "performance", "lab", "data"];
  const base = window.location.pathname.replace(/[^/]*$/, "");
  try {{
    pages.forEach(p => window.localStorage.setItem("stActiveTheme-" + base + p + "-v2", JSON.stringify("{theme_request}")));
  }} catch (e) {{ return; }}
  let target = window;
  try {{ if (window.top.location.host === window.location.host) target = window.top; }} catch (e) {{}}
  target.location.reload();
}})();
</script>""", unsafe_allow_javascript=True)
