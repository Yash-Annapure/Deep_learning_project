"""Streamlit page for the prototype:  uv run streamlit run app.py

Pick a saved run (prototype_runs/*.pkl.gz, made by prototype.py or by an earlier live run) or choose a new place on the
satellite map (or by coordinates) and a start month: the page then fetches 12 months of Sentinel-1 radar from Earth
Engine and labels every pixel with the trained model. With OFFLINE=1 the page works on a laptop without Earth Engine or
the model weights: saved runs and the map work, a new place needs the server. Edit this file and the browser updates on save.
"""
import datetime as dt
import glob
import os
import time

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import prototype as P

LIVE = os.environ.get("OFFLINE") != "1"
st.set_page_config(page_title="Sentinel-1 land cover over 12 months", layout="wide")


@st.cache_resource
def model():
    P.init_ee()
    return P.load_model()


@st.cache_resource
def load(path):
    """A saved run in two versions: the model's raw monthly maps, and smoothed over months (the 3-month mean the run
    saved, or P.despike for runs saved before it)."""
    res = P.load(path)
    res.update(P.second_opinion(res["maps"], res["wc"]))    # runs saved before a rule change get the current comparison
    clean = res.get("smooth") or P.despike(res["maps"])
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
    """Full-width table with one decimal, and a dash where there is no value (no radar, or nothing to compare with).
    As text, because the table would show an empty number as "None"; right-aligned like numbers."""
    st.dataframe(df.map(lambda v: "–" if pd.isna(v) else f"{v:.1f}"), width="stretch",
                 column_config={c: st.column_config.TextColumn(alignment="right") for c in df.columns})


NAMES = [c.replace("_", " ") for c in P.CLASSES]
WEAK = {"flooded vegetation", "bare"}          # low reliability on the holdout (IoU 0.07 and 0.24): marked with *
LEGEND = (" &nbsp; ".join(f"<span style='color:{c}; font-size:1.3em'>■</span> {n}{'*' * (n in WEAK)}"
                          for c, n in zip(P.COLORS, NAMES))
          + " &nbsp; <span style='color:#9ca3af'>* low reliability in testing: read these colours with care</span>")
START = (-0.0917, 34.7680)                    # the first place offered: Kisumu, Lake Victoria
MAP_PX = 600                                  # height of the satellite view (the picture view fills its half of the page)
SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
PLACES = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
CHANGED_HELP = ("How much of the square has a different class than in the previous month with radar. It counts every "
                "pixel that moved, so it can be large while the shares barely move (1% trees → crops and 1% crops → "
                "trees: 2% of pixels changed, 0 points). Part of it is noise: in our tests, with flicker removal on, up "
                "to about 1 in 10 pixels that did not really change still flipped in a month. Real change is large, "
                "clustered (switch on the magenta highlight to see where) and lasts more than one month.")


def trend_chart(shares):
    """Share of the square per class and month; hovering a month draws a rule and lists all classes' shares there.
    Months without radar are skipped, not drawn as gaps."""
    long = shares.dropna().assign(date=lambda d: pd.to_datetime(d.index + "-01")).melt("date", var_name="cls", value_name="share")
    base = alt.Chart(long).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0, tickCount="month")))
    lines = base.mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=36)).encode(
        y=alt.Y("share:Q", title="% of the square"),
        color=alt.Color("cls:N", title=None, scale=alt.Scale(domain=NAMES, range=P.COLORS), legend=alt.Legend(orient="bottom")))
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["date"], empty=False)
    rule = base.transform_pivot("cls", value="share", groupby=["date"]).mark_rule(color="#a3a3a3", strokeWidth=1.5).encode(
        opacity=alt.condition(hover, alt.value(0.9), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="month", format="%B %Y")]
                + [alt.Tooltip(field=n, type="quantitative", format=".1f", title=f"{n} (%)") for n in NAMES]).add_params(hover)
    return (lines + rule).properties(height=320)


def square(lat, lon):
    """The analysed 10 km square as [[south, west], [north, east]]. ponytail: the run's grid is a square in the local
    UTM projection, drawn as a latitude/longitude box; a few pixels off at the corners, fine for checking by eye."""
    h = P.SIZE_PX * P.SCALE / 2 / 111_320                         # half the side in degrees of latitude
    w = h / np.cos(np.radians(lat))
    return [[lat - h, lon - w], [lat + h, lon + w]]


def satellite_map(lat, lon, zoom):
    """Esri World Imagery (free for non-commercial use with this credit), place names on top."""
    m = folium.Map([lat, lon], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(SATELLITE, attr="Imagery © Esri, Maxar, Earthstar Geographics").add_to(m)
    folium.TileLayer(PLACES, attr="Esri", overlay=True).add_to(m)
    return m


def pick_place(shown=None):
    """Satellite map for a new place: a click moves the yellow 10 km square there and fills in the coordinates in the
    side panel; white dashed: the square of the results shown (shown = that run). The map itself is built the same way
    every time (only the squares change), so zoom and position stay."""
    box = folium.FeatureGroup()
    if shown:
        folium.Rectangle(square(shown["lat"], shown["lon"]), color="white", weight=2, dash_array="8", fill=False,
                         tooltip="the results shown below").add_to(box)
    folium.Rectangle(square(st.session_state.lat, st.session_state.lon), color="#ffd60a", weight=3, fill=False,
                     tooltip="the square to analyse (Run in the side panel)").add_to(box)
    out = st_folium(satellite_map(*START, 11), key="picker", height=450, use_container_width=True,
                    center=(st.session_state.lat, st.session_state.lon), feature_group_to_add=box,
                    returned_objects=["last_clicked"])                # panning and zooming do not rerun the page
    click = (out or {}).get("last_clicked")
    if click and click != st.session_state.get("clicked"):
        st.session_state.clicked = click
        st.session_state.lat = round(min(max(click["lat"], -60.0), 75.0), 4)   # the coordinate inputs' range
        st.session_state.lon = round((click["lng"] + 180) % 360 - 180, 4)      # the map repeats sideways: back to ±180
        st.rerun()                                                  # redraw the square at the new place


def overlay_map(res, k, show_change, opacity):
    """Month k's map laid over the satellite photo of the same square, at the chosen opacity (white line: the square)."""
    sq = square(res["lat"], res["lon"])
    box = folium.FeatureGroup()
    img = P.month_image(res, k, show_change)
    if img is not None:
        alpha = np.where(res["maps"][k] == P.NO_DATA, 0, 255).astype(np.uint8)   # no radar: see-through
        folium.raster_layers.ImageOverlay(np.dstack([img, alpha]), sq, opacity=opacity / 100).add_to(box)
    folium.Rectangle(sq, color="white", weight=2, fill=False).add_to(box)
    st_folium(satellite_map(res["lat"], res["lon"], 13), key="overlay", height=MAP_PX, use_container_width=True,
              feature_group_to_add=box, returned_objects=[], pixelated=True)
    st.caption(f"{res['months'][k]}" + ("" if img is not None else ": no full radar coverage this month") + ". The photo "
               "is a mosaic from recent years, not from these months, so a seasonal lake or a field can look different.")


@st.fragment
def month_viewer(res):
    """The 12 maps behind a slider, as a picture or over the satellite photo. Only this part of the page reruns when
    its controls move, so scrubbing stays quick."""
    changed = [P.changed_share(res, k) for k in range(12)]       # NaN for the first month with radar and months without
    busiest = max((k for k in range(12) if changed[k] == changed[k]), key=lambda k: changed[k],
                  default=next(i for i, mp in enumerate(res["maps"]) if mp is not None))   # opens on the month that changed most
    left, right = st.columns([1, 1])
    with right:
        month = st.select_slider("Month", res["months"], value=res["months"][busiest],
                                 help="Opens on the month whose map changed most since the month before.")
        show_change = st.toggle("Highlight pixels that changed since the previous month (magenta)")
        on_sat = st.toggle("Lay the map over the satellite photo",
                           help="Check *what is where*: villages, water, fields, forest. The photo is from another date, "
                                "so it shows the place, not the month's changes.")
        opacity = st.slider("Map opacity", 0, 100, 60, 5, format="%d%%", disabled=not on_sat,
                            help="0% shows only the satellite photo, 100% only the land-cover map.")
        play = st.button("Play the 12 months", icon=":material/play_arrow:", disabled=on_sat,
                         help="Shows the months one after another. Not over the satellite photo: it is one picture from "
                              "another date, so it would not change with them.")
        i = res["months"].index(month)
        share = P.changed_share(res, i)
        st.metric("Pixels that changed class since the previous month", "–" if share != share else f"{share:.1f}%",
                  help=CHANGED_HELP)
        if share != share and res["maps"][i] is not None:
            st.caption("The first month with radar: there is no earlier month to compare with, so no changes yet. "
                       "Move the slider to a later month.")
        table(pd.DataFrame({"share (%)": res["shares"][i], "change (points)": res["change"][i]}, index=NAMES))
        st.caption("**share**: % of the square in each class this month; the column adds up to 100, and 1% = 1 km². "
                   "**change**: points gained (+) or lost (−) since the previous month with radar; the column adds up to "
                   "0, because area only moves between classes.")
    with left:
        spot = st.empty()
        st.markdown(LEGEND, unsafe_allow_html=True)
    if on_sat:
        with spot.container():
            overlay_map(res, i, show_change, opacity)
        return

    def show(k):
        img = P.month_image(res, k, show_change)
        if img is None:
            spot.info(f"{res['months'][k]}: no full radar coverage this month")
        else:
            spot.image(img, caption=f"{res['months'][k]} · {res['scenes'][k]} radar images averaged (more images, less "
                                    "speckle)", width="stretch")              # fills its half of the page

    for k in (range(12) if play else [i]):
        show(k)
        if play:
            time.sleep(0.8)


# ---- 1. choose a place: on the map or by coordinates (a new run), or one saved earlier ----
st.title("Land cover from Sentinel-1 radar, month by month")
saved = sorted(glob.glob(f"{P.RUNS}/*.pkl.gz"), key=lambda p: load(p)[False].get("place", p))   # by name, as listed


def show_saved():
    """A saved place was picked: show it below, and move the map's square there."""
    run = load(st.session_state.saved_pick)[False]
    st.session_state.path = st.session_state.saved_pick
    st.session_state.lat, st.session_state.lon = round(run["lat"], 4), round(run["lon"], 4)


if "path" not in st.session_state:                                # first visit: the first saved place, if there is one
    st.session_state.path = saved[0] if saved else None
    if saved:
        st.session_state.saved_pick = saved[0]                    # the dropdown names what is shown
    first = load(saved[0])[False] if saved else dict(zip(("lat", "lon"), START))
    st.session_state.lat, st.session_state.lon = round(first["lat"], 4), round(first["lon"], 4)
st.header("1. Choose a place")
st.caption("Click the map to move the **yellow** 10 km square, then press Run in the side panel. The **white dashed** "
           "square is the place whose results are shown below.")
pick_place(load(st.session_state.path)[False] if st.session_state.path else None)   # before the inputs, so a click can fill them in
with st.sidebar:
    today = dt.date.today()
    last_full = today.year * 12 + today.month - 2                 # last complete month, counted in months since year 0
    default = last_full - 11                                      # the most recent 12 complete months: the best coverage
    st.header("Analyse a place")
    st.markdown("Click the map to move the yellow 10 km square, or type its centre. Then choose the first of the 12 "
                "months and press Run. A new place takes about a minute.")
    with st.form("inputs", border=False):
        lat = st.number_input("Latitude", -60.0, 75.0, key="lat", format="%.4f", help="South is negative.")
        lon = st.number_input("Longitude", -180.0, 180.0, key="lon", format="%.4f", help="West is negative.")
        a, b = st.columns(2)
        years = list(range(2017, today.year + 1))
        year = a.selectbox("Start year", years, index=years.index(default // 12))
        month = b.selectbox("Start month", list(range(1, 13)), index=default % 12,
                            format_func=lambda m: dt.date(2000, m, 1).strftime("%B"),
                            help="The first of the 12 months. The default is the most recent 12 complete months.")
        go = st.form_submit_button("Run for this square", type="primary")
    if go:
        if year * 12 + month - 1 + 11 > last_full:
            st.error("The 12 months must already be over. Pick an earlier start month.")
        elif not LIVE:
            st.warning("A new place needs Earth Engine and the model, so it runs on the server, not on this laptop "
                       "(OFFLINE=1). Here: open a saved place, or use the map to find places for later.")
        else:
            bar = st.progress(0.0, "Fetching 12 months of radar from Earth Engine...")
            try:
                st.session_state.path = st.session_state.saved_pick = compute(round(lat, 4), round(lon, 4),
                                                                              f"{year}-{month:02d}", bar)
                st.rerun()                                        # again, so the new run is in the dropdown and named there
            except ValueError as e:
                bar.empty()
                st.error(str(e))
    if saved:
        st.selectbox("…or open a place saved earlier", saved, key="saved_pick", on_change=show_saved,
                     format_func=lambda p: load(p)[False].get("place", os.path.basename(p)))

# ---- 2. the answer: what changed, then the months to look at ----
path = st.session_state.path
if path is None:
    st.info("Click the map above, then press Run in the side panel: the 12 months appear here.")
    st.stop()
st.divider()
card = st.container(border=True)                                  # the answer first: place, familiarity, what changed most
head, option = card.columns([3, 1], vertical_alignment="center")
clean = option.toggle("Remove one-month flickers", value=True,
                      help="Each month is averaged with the months before and after (the model's class probabilities, "
                           "where both have radar): a real change lasts at least 2 months and stays, radar speckle flips "
                           "a pixel for one month and goes. This halved the false changes in testing. A flood that lasts "
                           "one month goes too: switch this off to see it. Older saved runs use a simpler rule.")
res = load(path)[clean]
P.check(res)                                                      # shares add up to 100, changes to 0
have = [i for i, mp in enumerate(res["maps"]) if mp is not None]
head.header("2. " + res.get("place", f"{res['lat']:.4f}, {res['lon']:.4f}"))
sites = P.training_sites(res.get("weights", P.OLD_WEIGHTS))       # runs saved before this was recorded: exp4's model
card.caption(f"{res['lat']:.4f}, {res['lon']:.4f} · 10 km × 10 km · {res['months'][0]} to {res['months'][-1]} · "
             f"Sentinel-1 radar only, model trained on {len(sites)} sites")
km, near, what = P.familiarity(res["lat"], res["lon"], sites)
badge = ("green", "Close to the training data") if km < 100 else \
        ("blue", "In a region the model has learned from") if km < 500 else ("orange", "Far from all training data")
card.markdown(f":{badge[0]}-badge[{badge[1]}] &nbsp; Nearest training site: **{near.replace('_', ' ')}**"
              f"{f' ({what})' if what else ''}, {km:,.0f} km away."
              + (" The maps may be less reliable here; distance is only a rough guide, a far place can still look like "
                 "one the model knows." if km >= 500 else ""))
unknown = float(np.mean(res["wc"] == P.NO_DATA) * 100)           # WorldCover's snow/ice and moss: classes the model lacks
if unknown > 5:
    card.warning(f"ESA WorldCover marks {unknown:.0f}% of this square as snow, ice or tundra moss. The model has no class "
                 "for these (its training places had none), so those pixels get the nearest of its 7 classes: the maps "
                 "are unreliable there, whatever the distance to the training data says.", icon=":material/ac_unit:")
cols = card.columns(4)
last, first = res["months"][have[-1]], res["months"][have[0]]
for col, (c, pts) in zip(cols, P.biggest_changes(res)):
    i = P.CLASSES.index(c)
    s = res["shares"][have[-1]][i]
    col.metric(NAMES[i], f"{s:.1f}% of the area", f"{pts:+.1f} points since {first}", delta_color="off",
               help=f"{NAMES[i].capitalize()} covers {s:.1f}% of the square in {last}, i.e. {s:.1f} km² of 100 km². "
                    f"Below: the change since {first} in percentage points ({pts:+.1f} points = {abs(pts):.1f} km² "
                    f"{'more' if pts >= 0 else 'less'}). These are the classes whose share changed most over the 12 months.")
cols[3].metric("Months with radar", f"{len(have)} of 12",
               help="Months in which Sentinel-1 covered the whole square. A month without full coverage is left out, so a "
                    "gap is never counted as a change. Fewer months = a thinner picture of the year.")
picked = (round(st.session_state.lat, 4), round(st.session_state.lon, 4))
if picked != (round(res["lat"], 4), round(res["lon"], 4)):
    st.info(icon=":material/location_on:", body=f"The yellow square you picked ({picked[0]:.4f}, {picked[1]:.4f}) has not been analysed yet: press **Run "
            f"for this square** in the side panel. Below are the results for the white dashed square.")
with st.expander("How to read the numbers on this page"):
    st.markdown(f"""
- **% of the area**: how much of the 10 km × 10 km square a class covers. The square is 100 km², so **1% = 1 km²**.
  All classes add up to 100%.
- **points**: a change in that share. 20% → 25% is **+5 points**, i.e. 5 km² more. Area only moves between classes,
  so the changes of all classes add up to 0.
- **Pixels that changed class**: {CHANGED_HELP}
- **Months with radar**: months in which Sentinel-1 covered the whole square. The others are left out (grey), never
  counted as change.
- **Remove one-month flickers** (switch next to the heading): each month is averaged with its neighbours, so speckle that flips a pixel for
  a single month disappears; a change that lasts stays. Switch it off to see the raw monthly maps.
- **Low reliability (\\*)**: in the final test on 14 places the model never saw, it recognised flooded vegetation in only
  14% of cases and called 40% of bare ground built-up (dry, bright soil echoes like buildings in radar). Water,
  built-up areas and trees are reliable (IoU 0.79, 0.81 and 0.63); crops and low vegetation are in between (about 0.4).
- **Snow and ice**: the model has no class for them. Where WorldCover shows them, a warning appears above.
- **Agreement** (tab *Compared with other maps*): the share of pixels where two maps give the same class. No map is
  the truth: read the model's agreement next to how well the two optical maps agree with each other.
- **Nearest training site** (badge above): how far this square is from the closest place the model learned from.
  Farther means less familiar, so read the maps with more care.
""")
missing = [m for m, mp in zip(res["months"], res["maps"]) if mp is None]
if missing:
    st.warning(f"No full radar coverage in {', '.join(missing)}. Those months are left out, and changes are measured against "
               "the last month with data (a partly covered month would show changes that never happened).")

month_viewer(res)

trends, opinion, grid = st.tabs(["Trends", "Compared with other maps", "All 12 months"])
with trends:
    shares = pd.DataFrame(res["shares"], index=res["months"], columns=NAMES)
    st.subheader("Share of the area per class (%)")
    st.altair_chart(trend_chart(shares), width="stretch")
    st.caption("One line per class: its share of the square each month. A flat line is a stable class; a line that rises "
               "while another falls is area moving between those two classes (for example water → bare as a reservoir "
               "dries out, bare → crops as fields are planted).")
    table(shares)
    st.subheader("Change from the previous month (percentage points)")
    table(pd.DataFrame(res["change"], index=res["months"], columns=NAMES))
    st.subheader(f"Total change, {res['months'][have[0]]} to {res['months'][have[-1]]} (percentage points)")
    table(pd.DataFrame([res["total"]], columns=NAMES, index=["total"]))
    st.caption(("One-month flickers removed (switch next to the heading); the first and last month have no neighbour on one side and are "
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
        m1.metric("Model vs Dynamic World", f"{P.agreement(res['typical'], dw):.0%}",
                  help="How closely the model reproduces Dynamic World, the optical map it learned from, over the same 12 "
                       "months. 100% would mean radar alone gives the same map as cloud-free optical images.")
        m3.metric("Dynamic World vs WorldCover", f"{P.agreement(dw, res['wc']):.0%}",
                  help="How much two independent optical maps agree with each other here. A reference point: the model "
                       "cannot be expected to agree with WorldCover much more than this.")
    m2.metric("Model vs WorldCover", f"{res['agreement']:.0%}",
              help="How closely the model matches ESA WorldCover 2021, a second map it did not learn from. Read it next to "
                   "'Dynamic World vs WorldCover': if that is low too, the two optical maps disagree, not just the model.")
    st.caption(f"Pixels that agree, out of those both maps classify. Each map over the year: the most frequent class, but crops "
               f"if crops in {P.CROP_MONTHS} or more months (WorldCover's definition of cropland: planted at least once a year; "
               "a plain majority would call most fields bare or low vegetation, because they carry a crop only part of the year). "
               "None of the three is the truth: the model learned from Dynamic World, so it can at best match Dynamic World; "
               "WorldCover is from 2021, uses its own class definitions (trees from 10% canopy, for example) and filtered this "
               "model's training labels. Where Dynamic World and WorldCover disagree, the model cannot satisfy both.")
with grid:
    blank = np.full((250, 250, 3), 38, np.uint8)                  # months without radar: dark grey
    for row in range(3):                                          # a 4-column grid: the columns' gap separates the maps
        for col, k in zip(st.columns(4), range(4 * row, 4 * row + 4)):
            col.image(blank if res["maps"][k] is None else P.month_image(res, k)[::4, ::4],
                      caption=res["months"][k] + ("" if res["maps"][k] is not None else ": no full radar coverage"),
                      width="stretch")
    st.markdown(LEGEND, unsafe_allow_html=True)
