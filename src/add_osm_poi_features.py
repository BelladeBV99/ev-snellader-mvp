import json
import time
from pathlib import Path

import pandas as pd
import requests

IN_PATH = "data/mvp_train_enriched.csv"
OUT_PATH = "data/mvp_train_enriched_500m.csv"
CACHE_PATH = Path("data/osm_cache_v2.json")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

RADIUS_M = 500  # 500 m: lokale omgeving

# We beperken ons tot “lichte” en vaak relevante POI’s
# (motorway bewust weggelaten)
GROUPS = {
    "shop_any": ('["shop"]',),
    "amenity_food": ('["amenity"~"restaurant|cafe|fast_food|bar|pub"]',),
    "office_any": ('["office"]',),
    "parking": ('["amenity"="parking"]',),
    "fuel": ('["amenity"="fuel"]',),
}

def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}

def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache))

cache = load_cache()

def build_query(lat: float, lon: float) -> str:
    parts = []
    for name, selectors in GROUPS.items():
        sel = selectors[0]
        # out count per group: we gebruiken 'out count' met een set per group en lezen tags.total
        parts.append(f"""
        (
          node{sel}(around:{RADIUS_M},{lat},{lon});
          way{sel}(around:{RADIUS_M},{lat},{lon});
          relation{sel}(around:{RADIUS_M},{lat},{lon});
        )->.{name};
        """)

    # 'out count' per set:
    out_counts = "\n".join([f'.{name} out count;' for name in GROUPS.keys()])

    return f"""
    [out:json][timeout:120];
    {''.join(parts)}
    {out_counts}
    """

def fetch_counts(lat: float, lon: float) -> dict:
    key = f"{lat:.5f}|{lon:.5f}|r{RADIUS_M}"
    if key in cache:
        return cache[key]

    query = build_query(lat, lon)

    # retry met backoff
    backoff = 5
    for attempt in range(1, 8):  # max 7 pogingen
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=150)
            if resp.status_code in (429, 504):
                raise requests.HTTPError(f"{resp.status_code} throttled/timeout", response=resp)
            resp.raise_for_status()
            data = resp.json()

            # Overpass geeft meerdere 'count' elements terug, in dezelfde volgorde als out count
            counts = {}
            count_elems = [e for e in data.get("elements", []) if e.get("type") == "count"]
            for name, elem in zip(GROUPS.keys(), count_elems):
                counts[name] = int(elem.get("tags", {}).get("total", 0))

            cache[key] = counts
            return counts

        except Exception as e:
            if attempt == 7:
                # geef 0's terug als het echt niet lukt
                counts = {k: 0 for k in GROUPS.keys()}
                cache[key] = counts
                return counts
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

def main():
    df = pd.read_csv(IN_PATH)

    for k in GROUPS.keys():
        df[f"osm_{k}_{RADIUS_M}m"] = 0

    n = len(df)
    for i, row in df.iterrows():
        lat = float(row["evse_latitude"])
        lon = float(row["evse_longitude"])
        print(f"[{i+1}/{n}] OSM counts for {lat:.5f}, {lon:.5f}")

        counts = fetch_counts(lat, lon)
        for k, v in counts.items():
            df.at[i, f"osm_{k}_{RADIUS_M}m"] = v

        # beleefde pauze
        time.sleep(1.2)

        if (i + 1) % 10 == 0:
            save_cache(cache)

    save_cache(cache)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Opgeslagen: {OUT_PATH}")

if __name__ == "__main__":
    main()