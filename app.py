"""Streamlit page for the prototype:  uv run streamlit run app.py

Enter a location and a start month; the page fetches 12 months of Sentinel-1 radar from Earth Engine, labels every
pixel with the trained model, and shows the 12 maps, the share of each class per month, how it changed, and an
independent check against ESA WorldCover.
"""
import datetime as dt

import pandas as pd
import streamlit as st

import prototype as P

st.set_page_config(page_title="Sentinel-1 land cover over 12 months", layout="wide")
st.title("Land cover from Sentinel-1 radar, month by month")
st.caption("Radar sees through clouds, so this works in the rainy season too. Model: U-Net trained on 21 sites in Africa and Asia.")


@st.cache_resource
def setup():
    P.init_ee()
    return P.load_model()


@st.cache_data(show_spinner=False)
def compute(lat, lon, start):
    return P.run(lat, lon, start, model)


def table(df):
    """Full-width table with one decimal, and a dash where there is no value (no radar, or nothing to compare with)."""
    st.dataframe(df.style.format("{:.1f}", na_rep="–"), width="stretch")


model = setup()
today = dt.date.today()
last_full = (today.year * 12 + today.month - 2)               # last complete month, counted in months since year 0
with st.sidebar.form("inputs"):
    st.header("Location and time")
    lat = st.number_input("Latitude", -60.0, 75.0, -0.0917, format="%.4f")
    lon = st.number_input("Longitude", -180.0, 180.0, 34.7680, format="%.4f")
    year = st.selectbox("Start year", list(range(2017, today.year + 1)), index=2024 - 2017)
    month = st.selectbox("Start month", list(range(1, 13)), format_func=lambda m: dt.date(2000, m, 1).strftime("%B"))
    go = st.form_submit_button("Run")

if not go:
    st.info("Pick a location and a start month on the left, then press Run. A run takes about a minute.")
    st.stop()
if year * 12 + month - 1 + 11 > last_full:
    st.error("The 12 months must already be over. Pick an earlier start month.")
    st.stop()

start = f"{year}-{month:02d}"
with st.spinner("Fetching 12 months of radar from Earth Engine and labelling them..."):
    try:
        res = compute(round(lat, 4), round(lon, 4), start)
    except ValueError as e:
        st.error(str(e))
        st.stop()
P.check(res)                                                     # shares add up to 100, changes to 0

st.pyplot(P.maps_figure(res))
missing = [m for m, mp in zip(res["months"], res["maps"]) if mp is None]
if missing:
    st.warning(f"No full radar coverage in {', '.join(missing)}. Those months are left out, and changes are measured against "
               "the last month with data (a partly covered month would show changes that never happened).")

shares = pd.DataFrame(res["shares"], index=res["months"], columns=P.CLASSES)
st.subheader("Share of the area per class over the 12 months (%)")
st.line_chart(shares.dropna(), color=P.COLORS)                   # months without radar are skipped, not drawn as gaps
table(shares)

st.subheader("Change from the previous month (percentage points)")
table(pd.DataFrame(res["change"], index=res["months"], columns=P.CLASSES))
st.subheader(f"Total change, {res['months'][0]} to {res['months'][-1]} (percentage points)")
table(pd.DataFrame([res["total"]], columns=P.CLASSES, index=["total"]))
st.caption(f"{res['pass_dir'].capitalize()} passes, scenes per month: {res['scenes']}. Changes include classification "
           "errors and seasonal effects (for example wet versus dry season), not only real land-use change.")

st.header("Independent check: ESA WorldCover 2021")
st.pyplot(P.comparison_figure(res))
st.metric("Pixels where the model's typical map agrees with WorldCover", f"{res['agreement']:.0%}")
table(pd.DataFrame([res["typical_shares"], res["wc_shares"]], columns=P.CLASSES,
                   index=["model (each pixel's most frequent class)", "ESA WorldCover 2021"]))
st.caption("WorldCover was not used to train this model, so it is an independent second opinion, not the truth: it is from "
           "2021 (the land may have changed since), it uses its own class definitions, and it is partly made from Sentinel-1 too. "
           "For reference, Dynamic World (the training labels) and WorldCover agree on only about 60% of pixels at our sites.")
