import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from catboost import CatBoostRegressor
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError
from joblib import load
from pyproj import Transformer
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6371000.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_CACHE_PATH = Path("data/osm_cache_v2.json")

st.set_page_config(page_title="EV snellader voorspeller", layout="centered")
st.title("⚡ EV snellader – voorspelling laadsessies per dag")

# -----------------------
# Cached loaders (heavy files)
# -----------------------

@st.cache_resource
def load_model_and_meta():
    model = CatBoostRegressor()
    model.load_model("models/final_ev_sessions_model.cbm")
    meta = load("models/final_ev_sessions_meta.joblib")
    return model, meta

@st.cache_resource
def load_pop_map():
    pop = pd.read_csv("data/statbel_population_grid_1km_2020.csv")
    return dict(zip(pop["GRD_NEWID"], pop["MS_POPULATION_20200101"]))

@st.cache_resource
def load_competitor_tree():
    comp = pd.read_csv("data/fast_chargers_be_osm.csv")
    coords = np.deg2rad(comp[["lat", "lon"]].values.astype(float))
    return BallTree(coords, metric="haversine")

# -----------------------
# OSM cache loader (NOT cached, because we write to it)
# -----------------------

def load_osm_cache_and_path():
    if not OSM_CACHE_PATH.exists():
        return {}, OSM_CACHE_PATH
    try:
        return json.loads(OSM_CACHE_PATH.read_text()), OSM_CACHE_PATH
    except Exception:
        return {}, OSM_CACHE_PATH

# -----------------------
# Feature helpers
# -----------------------

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

def population_1km(lat: float, lon: float, pop_map: dict) -> int:
    x, y = _transformer.transform(lon, lat)
    e_km = int(x // 1000)
    n_km = int(y // 1000)
    gid = f"1kmN{n_km}E{e_km}"
    return int(pop_map.get(gid, 0))

def competitors_fast_500m(lat: float, lon: float, tree: BallTree) -> int:
    r_rad = 500 / EARTH_RADIUS_M
    pt = np.deg2rad(np.array([[lat, lon]], dtype=float))
    return int(tree.query_radius(pt, r=r_rad, count_only=True)[0])

def overpass_office_shop_counts_500m(lat: float, lon: float) -> tuple[int, int]:
    query = f"""
    [out:json][timeout:90];
    (
      node["office"](around:500,{lat},{lon});
      way["office"](around:500,{lat},{lon});
      relation["office"](around:500,{lat},{lon});
    )->.office_any;

    (
      node["shop"](around:500,{lat},{lon});
      way["shop"](around:500,{lat},{lon});
      relation["shop"](around:500,{lat},{lon});
    )->.shop_any;

    .office_any out count;
    .shop_any out count;
    """

    backoff = 5
    for attempt in range(1, 7):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
            if r.status_code in (429, 504):
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            data = r.json()

            counts = [e for e in data.get("elements", []) if e.get("type") == "count"]
            if len(counts) >= 2:
                office = int(counts[0].get("tags", {}).get("total", 0))
                shop = int(counts[1].get("tags", {}).get("total", 0))
                return office, shop

            return 0, 0

        except Exception:
            if attempt == 6:
                return 0, 0
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

def osm_counts_500m(lat: float, lon: float, cache: dict, cache_path: Path) -> tuple[int, int, bool]:
    key = f"{lat:.5f}|{lon:.5f}|r500"

    if key in cache:
        v = cache.get(key, {})
        return int(v.get("office_any", 0)), int(v.get("shop_any", 0)), True

    office_cnt, shop_cnt = overpass_office_shop_counts_500m(lat, lon)

    cache[key] = {"office_any": int(office_cnt), "shop_any": int(shop_cnt)}
    try:
        cache_path.write_text(json.dumps(cache))
    except Exception:
        pass

    return int(office_cnt), int(shop_cnt), False

# -----------------------
# Load assets
# -----------------------

model, meta = load_model_and_meta()
features = meta["features"]

# optional: if you added this in training later
cat_levels = meta.get("cat_levels", {})

pop_map = load_pop_map()
competitor_tree = load_competitor_tree()

@st.cache_resource
def get_geocoders():
    nom = Nominatim(user_agent="bellade-ev-sessions/1.0", timeout=10)
    arc = ArcGIS(timeout=10)
    return nom, arc

nominatim, arcgis = get_geocoders()

def geocode_address(address: str):
    query = address.strip() + ", Belgium"

    # 1) Nominatim retries
    backoff = 1.5
    for _ in range(3):
        try:
            loc = nominatim.geocode(query)
            if loc:
                return float(loc.latitude), float(loc.longitude), "Nominatim"
            break
        except (GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError):
            time.sleep(backoff)
            backoff *= 2

    # 2) ArcGIS fallback retries
    backoff = 1.5
    for _ in range(3):
        try:
            loc = arcgis.geocode(query)
            if loc:
                return float(loc.latitude), float(loc.longitude), "ArcGIS"
            break
        except (GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError):
            time.sleep(backoff)
            backoff *= 2

    return None, None, None
# -----------------------
# UI
# -----------------------

# SiteType options: prefer meta values if they exist, else fallback
default_site_types = ["Store", "Office building", "Parking area", "Unknown"]
site_type_options = cat_levels.get("Pool_SiteType", default_site_types)
site_type = st.selectbox("Site type", site_type_options)

# New optional dropdowns: only show if the model expects them
oprit_value = None
trafic_value = None
if "Trafic" in features:
    # Bepaal min/max op basis van je training data (veilig fallback)
    try:
        df_ref = pd.read_csv("data/mvp_train_enriched_500m_comp.csv")
        t_min = int(pd.to_numeric(df_ref["Trafic"], errors="coerce").dropna().min())
        t_max = int(pd.to_numeric(df_ref["Trafic"], errors="coerce").dropna().max())
        t_med = int(pd.to_numeric(df_ref["Trafic"], errors="coerce").dropna().median())
    except Exception:
        t_min, t_max, t_med = 0, 20, 10  # fallback
        

    trafic_value = st.slider(
        "Trafic",
        min_value=t_min,
        max_value=t_max,
        value=t_med,
        step=1,
        help="Numerieke verkeersscore (hoger = drukker)."
    )

if "Oprit" in features:
    oprit_options = cat_levels.get("Oprit", ["Yes", "No", "Unknown"])
    oprit_value = st.selectbox("Oprit", oprit_options)

address = st.text_input("Adres (België)", placeholder="Bijv. Berlaarsestraat 65, Lier")

if st.button("Voorspel"):
    if not address.strip():
        st.error("Vul een adres in.")
        st.stop()

lat, lon, provider = geocode_address(address)

if lat is None:
    st.error(
        "Adres kon niet opgezocht worden (geocoder tijdelijk onbereikbaar). "
        "Probeer opnieuw of geef straat + nummer + gemeente."
    )
    st.stop()

st.caption(f"Geocoding via: {provider}")

    lat, lon = float(loc.latitude), float(loc.longitude)

    osm_cache, osm_cache_path = load_osm_cache_and_path()

    pop = population_1km(lat, lon, pop_map)
    comp = competitors_fast_500m(lat, lon, competitor_tree)
    office_cnt, shop_cnt, from_cache = osm_counts_500m(lat, lon, osm_cache, osm_cache_path)

    # Base features always present in your model
    row = {
        "evse_latitude": lat,
        "evse_longitude": lon,
        "Pool_SiteType": site_type,
        "population_1km": pop,
        "osm_office_any_500m": office_cnt,
        "osm_shop_any_500m": shop_cnt,
        "competitors_fast_500m": comp,
    }

    # Add optional new features if required by the trained model
    if "Trafic" in features:
        row["Trafic"] = trafic_value if trafic_value is not None else "Unknown"
    if "Oprit" in features:
        row["Oprit"] = oprit_value if oprit_value is not None else "Unknown"

    # Align columns exactly to training feature order
    X = pd.DataFrame([row])
    for col in features:
        if col not in X.columns:
            # safe fallback if training expects a column not set in row
            X[col] = "Unknown" if col in ("Pool_SiteType", "Trafic", "Oprit") else 0.0
    X = X[features]

    pred = float(model.predict(X)[0])

    st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))

    st.subheader("Resultaat")
    st.metric("Voorspelde laadsessies per dag", f"{pred:.1f}")

    st.write("📍 Details")
    st.write(f"- Latitude: {lat:.6f}")
    st.write(f"- Longitude: {lon:.6f}")
    st.write(f"- Bevolking (1 km²): {pop}")
    st.write(f"- Kantoren (OSM) binnen 500m: {office_cnt}")
    st.write(f"- Shops (OSM) binnen 500m: {shop_cnt}")

    # Only show if used
    if "Trafic" in features:
        st.write(f"- Trafic: {row.get('Trafic')}")
    if "Oprit" in features:
        st.write(f"- Oprit: {row.get('Oprit')}")

    if not from_cache:
        st.info("OSM office/shop werden live opgehaald (cache-miss) en zijn nu gecached voor volgende keer.")