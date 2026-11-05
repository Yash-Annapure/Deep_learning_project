"""Streamlit page for the prototype:  uv run streamlit run app.py

Pick a saved run (prototype_runs/*.pkl.gz, made by prototype.py or by an earlier live run) or enter a new location and
a start month: the page then fetches 12 months of Sentinel-1 radar from Earth Engine and labels every pixel with the
trained model. With OFFLINE=1 only saved runs are offered, so the page works on a laptop without Earth Engine or the
model weights: edit this file and the browser updates on save.
"""
import datetime as dt
import glob
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import prototype as P

LIVE = os.environ.get("OFFLINE") != "1"
st.set_page_config(page_title="Sentinel-1 land cover over 12 months", layout="wide")


@st.cache_resource
def model():
    P.init_ee()
    return P.load_model()


@st.cache_resource
def load(path):
    """A saved run in two versions: the model's raw monthly maps, and with one-month flickers removed (P.despike)."""
    res = P.load(path)
    res.update(P.second_opinion(res["maps"], res["wc"]))    # runs saved before a rule change get the current comparison
    clean = P.despike(res["maps"])
    return {True: {**res, "maps": clean, **P.summarise(clean)}, False: res}


def compute(lat, lon, start, bar):
    """A live run, saved like the demo runs, so asking again for the same place and months is instant."""
    path = f"{P.RUNS}/{lat:.4f}_{lon:.4f}_{start}.pkl.gz"
    if not os.path.exists(path):
        res = P.run(lat, lon, start, model(), progress=lambda n: bar.progress(n / 12, f"{n} of 12 months of radar fetched"))
        res["place"] = f"{lat:.4f}, {lon:.4f}"
        os.makedirs(P.RUNS, exist_ok=True)
        P.save(res, path)
    return path


def table(df):
    """Full-width table with one decimal, and a dash where there is no value (no radar, or nothing to compare with)."""
    st.dataframe(df.style.format("{:.1f}", na_rep="–"), width="stretch")


NAMES = [c.replace("_", " ") for c in P.CLASSES]
LEGEND = " &nbsp; ".join(f"<span style='color:{c}; font-size:1.3em'>■</span> {n}" for c, n in zip(P.COLORS, NAMES))


@st.fragment
def month_viewer(res):
    """The 12 maps behind a slider. Only this part of the page reruns when the slider moves, so scrubbing stays quick."""
    first = next(i for i, mp in enumerate(res["maps"]) if mp is not None)
    left, right = st.columns([3, 2])
    with right:
        month = st.select_slider("Month", res["months"], value=res["months"][first])
        show_change = st.toggle("Highlight pixels that changed since the previous month (magenta)")
        play = st.button("▶ Play the 12 months")
        i = res["months"].index(month)
        share = P.changed_share(res, i)
        st.metric("Pixels that changed class since the previous month", "–" if share != share else f"{share:.1f}%")
        table(pd.DataFrame({"share (%)": res["shares"][i], "change (points)": res["change"][i]}, index=NAMES))
    with left:
        spot = st.empty()
        st.markdown(LEGEND, unsafe_allow_html=True)

    def show(k):
        img = P.month_image(res, k, show_change)
        if img is None:
            spot.info(f"{res['months'][k]}: no full radar coverage this month")
        else:
            spot.image(img, caption=f"{res['months'][k]}  ({res['scenes'][k]} scenes)", width="stretch")

    for k in (range(12) if play else [i]):
        show(k)
        if play:
            time.sleep(0.8)


# ---- where the result comes from: a saved run, or a new location fetched live ----
saved = sorted(glob.glob(f"{P.RUNS}/*.pkl.gz"))
choices = (["Saved run"] if saved else []) + (["New location"] if LIVE else [])
if not choices:
    st.error(f"No saved runs in {P.RUNS}/ and OFFLINE=1: copy some .pkl.gz files there first.")
    st.stop()
mode = st.sidebar.radio("Show", choices)
if mode == "Saved run":
    path = st.sidebar.selectbox("Place", saved, format_func=lambda p: load(p)[False].get("place", os.path.basename(p)))
else:
    today = dt.date.today()
    last_full = today.year * 12 + today.month - 2                 # last complete month, counted in months since year 0
    default = last_full - 11                                      # the most recent 12 complete months: the best coverage
    with st.sidebar.form("inputs"):
        lat = st.number_input("Latitude", -60.0, 75.0, -0.0917, format="%.4f")
        lon = st.number_input("Longitude", -180.0, 180.0, 34.7680, format="%.4f")
        years = list(range(2017, today.year + 1))
        year = st.selectbox("Start year", years, index=years.index(default // 12))
        month = st.selectbox("Start month", list(range(1, 13)), index=default % 12,
                             format_func=lambda m: dt.date(2000, m, 1).strftime("%B"))
        go = st.form_submit_button("Run")
    if go:                                                        # remembered, so touching anything later keeps the results
        st.session_state.last_run = (lat, lon, year, month)
    if "last_run" not in st.session_state:
        st.title("Land cover from Sentinel-1 radar, month by month")
        st.info("Enter a location and a start month on the left, then press Run. A new place takes about a minute.")
        st.stop()
    lat, lon, year, month = st.session_state.last_run
    if year * 12 + month - 1 + 11 > last_full:
        st.error("The 12 months must already be over. Pick an earlier start month.")
        st.stop()
    bar = st.progress(0.0, "Fetching 12 months of radar from Earth Engine...")
    try:
        path = compute(round(lat, 4), round(lon, 4), f"{year}-{month:02d}", bar)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    bar.empty()

clean = st.sidebar.toggle("Remove one-month flickers", value=True,
                          help="A pixel whose class differs from both the month before and the month after, which agree, "
                               "takes their class: a real change lasts at least 2 months, radar speckle flips a pixel for one.")
res = load(path)[clean]
P.check(res)                                                      # shares add up to 100, changes to 0
have = [i for i, mp in enumerate(res["maps"]) if mp is not None]

# ---- the answer first: what changed, then the months to look at ----
st.title(res.get("place", f"{res['lat']:.4f}, {res['lon']:.4f}"))
sites = P.training_sites(res.get("weights", P.WEIGHTS))           # runs saved before this was recorded: the default model
st.caption(f"{res['lat']:.4f}, {res['lon']:.4f} · 10 km × 10 km · {res['months'][0]} to {res['months'][-1]} · "
           f"Sentinel-1 radar only, model trained on {len(sites)} sites")
km, near, what = P.familiarity(res["lat"], res["lon"], sites)
nearest = f"Nearest training site: **{near.replace('_', ' ')}**{f' ({what})' if what else ''}, {km:,.0f} km away."
if km < 100:
    st.success(f"Close to the training data. {nearest}")
elif km < 500:
    st.info(f"In a region the model has learned from. {nearest}")
else:
    st.warning(f"Far from all training data, so the maps may be less reliable here. {nearest} "
               "Distance is only a rough guide: a far place can still look like one the model knows.")
cols = st.columns(4)
for col, (c, pts) in zip(cols, P.biggest_changes(res)):
    i = P.CLASSES.index(c)
    col.metric(NAMES[i], f"{res['shares'][have[-1]][i]:.1f}% of the area",
               f"{pts:+.1f} points since {res['months'][have[0]]}", delta_color="off")
cols[3].metric("Months with radar", f"{len(have)} of 12")
missing = [m for m, mp in zip(res["months"], res["maps"]) if mp is None]
if missing:
    st.warning(f"No full radar coverage in {', '.join(missing)}. Those months are left out, and changes are measured against "
               "the last month with data (a partly covered month would show changes that never happened).")

month_viewer(res)

trends, opinion, grid = st.tabs(["Trends", "Compared with other maps", "All 12 months"])
with trends:
    shares = pd.DataFrame(res["shares"], index=res["months"], columns=NAMES)
    st.subheader("Share of the area per class (%)")
    st.line_chart(shares.dropna(), color=P.COLORS)               # months without radar are skipped, not drawn as gaps
    table(shares)
    st.subheader("Change from the previous month (percentage points)")
    table(pd.DataFrame(res["change"], index=res["months"], columns=NAMES))
    st.subheader(f"Total change, {res['months'][have[0]]} to {res['months'][have[-1]]} (percentage points)")
    table(pd.DataFrame([res["total"]], columns=NAMES, index=["total"]))
    st.caption(("One-month flickers removed (sidebar); the first and last month have no neighbour on one side and are "
                "kept as they are, so the total change is the same either way. " if clean else "Raw monthly maps. ")
               + f"{res['pass_dir'].capitalize()} passes, scenes per month: {res['scenes']}. Changes include classification "
               "errors and seasonal effects (for example wet versus dry season), not only real land-use change.")
with opinion:
    dw = res.get("dw")                                            # runs saved before Dynamic World was added have none
    a, b, c = st.columns(3)
    a.image(P.LUT[res["typical"]], caption="this model (radar only)", width="stretch")
    if dw is None:
        b.info("Dynamic World: not in this saved run (run the place again to get it)")
    else:
        b.image(P.LUT[dw], caption="Dynamic World, same 12 months (optical, the model's teacher)", width="stretch")
    c.image(P.LUT[res["wc"]], caption="ESA WorldCover 2021 (optical + radar)", width="stretch")
    st.markdown(LEGEND, unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    if dw is not None:
        m1.metric("Model vs Dynamic World", f"{P.agreement(res['typical'], dw):.0%}", help="how well the model learned from its teacher")
        m3.metric("Dynamic World vs WorldCover", f"{P.agreement(dw, res['wc']):.0%}", help="how much the two optical maps agree with each other")
    m2.metric("Model vs WorldCover", f"{res['agreement']:.0%}")
    st.caption(f"Pixels that agree, out of those both maps classify. Each map over the year: the most frequent class, but crops "
               f"if crops in {P.CROP_MONTHS} or more months (WorldCover's definition of cropland: planted at least once a year; "
               "a plain majority would call most fields bare or low vegetation, because they carry a crop only part of the year). "
               "None of the three is the truth: the model learned from Dynamic World, so it can at best match Dynamic World; "
               "WorldCover is from 2021, uses its own class definitions (trees from 10% canopy, for example) and filtered this "
               "model's training labels. Where Dynamic World and WorldCover disagree, the model cannot satisfy both.")
with grid:
    blank = np.full((250, 250, 3), 230, np.uint8)                 # months without radar: grey
    imgs = [blank if res["maps"][k] is None else P.month_image(res, k)[::4, ::4] for k in range(12)]
    st.image(imgs, caption=res["months"], width=220)
    st.markdown(LEGEND, unsafe_allow_html=True)
