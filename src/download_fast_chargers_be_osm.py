import time
import json
from pathlib import Path

import pandas as pd
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUT = Path("data/fast_chargers_be_osm.csv")
CACHE = Path("data/fast_chargers_be_osm_cache.json")

# België bbox (ruim genomen)
# (south, west, north, east)
BBOX = (49.45, 2.50, 51.60, 6.45)

# tegelgrootte in graden (kleiner = minder timeouts, meer calls)
STEP = 0.6

def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}

def save_cache(cache):
    CACHE.write_text(json.dumps(cache))

cache = load_cache()

def overpass_tile(south, west, north, east):
    key = f"{south:.2f},{west:.2f},{north:.2f},{east:.2f}"
    if key in cache:
        return cache[key]

    # "Fast" heuristic:
    # - charging_station met DC-connector tags (ccs/chademo) OF expliciete output tags.
    # We nemen nodes + ways + relations, en vragen tags + center-coords.
    query = f"""
    [out:json][timeout:180];
    (
      node["amenity"="charging_station"]({south},{west},{north},{east});
      way["amenity"="charging_station"]({south},{west},{north},{east});
      relation["amenity"="charging_station"]({south},{west},{north},{east});
    );
    out tags center;
    """

    backoff = 5
    for attempt in range(1, 8):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=240)
            if r.status_code in (429, 504):
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            data = r.json()
            cache[key] = data
            return data
        except Exception as e:
            if attempt == 7:
                cache[key] = {"elements": []}
                return cache[key]
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

def is_fast(tags: dict) -> bool:
    # DC connector hints
    dc_hint = any(
        k in tags for k in [
            "socket:ccs", "socket:chademo", "socket:ccs2", "socket:tesla_supercharger"
        ]
    )
    # Output hints (als ingevuld)
    # voorbeelden: charging_station:output=150 kW, socket:ccs:output=150
    output_keys = [k for k in tags.keys() if k.endswith(":output") or k == "charging_station:output"]
    output_kw = 0.0
    for k in output_keys:
        v = str(tags.get(k, "")).lower().replace("kw", "").strip()
        try:
            output_kw = max(output_kw, float(v))
        except:
            pass

    # "fast" = ≥ 50kW indien output bekend; anders DC-hint
    if output_kw >= 50:
        return True
    if output_kw == 0 and dc_hint:
        return True
    return False

def get_lat_lon(el):
    if el["type"] == "node":
        return el.get("lat"), el.get("lon")
    c = el.get("center", {})
    return c.get("lat"), c.get("lon")

def main():
    south, west, north, east = BBOX

    tiles = []
    s = south
    while s < north:
        n = min(s + STEP, north)
        w = west
        while w < east:
            e = min(w + STEP, east)
            tiles.append((s, w, n, e))
            w += STEP
        s += STEP

    rows = []
    seen = set()

    for i, (s, w, n, e) in enumerate(tiles, start=1):
        print(f"[{i}/{len(tiles)}] tile {s:.2f},{w:.2f},{n:.2f},{e:.2f}")
        data = overpass_tile(s, w, n, e)
        for el in data.get("elements", []):
            el_id = f'{el.get("type")}:{el.get("id")}'
            if el_id in seen:
                continue
            seen.add(el_id)

            tags = el.get("tags", {})
            if not is_fast(tags):
                continue

            lat, lon = get_lat_lon(el)
            if lat is None or lon is None:
                continue

            rows.append({
                "osm_id": el_id,
                "lat": float(lat),
                "lon": float(lon),
                "name": tags.get("name", ""),
                "operator": tags.get("operator", ""),
                "network": tags.get("network", ""),
            })

        # af en toe cache wegschrijven
        if i % 5 == 0:
            save_cache(cache)
        time.sleep(1.0)

    save_cache(cache)

    df = pd.DataFrame(rows).drop_duplicates(subset=["osm_id"])
    df.to_csv(OUT, index=False)
    print(f"\n✅ Saved fast chargers: {OUT} (n={len(df)})")

if __name__ == "__main__":
    main()