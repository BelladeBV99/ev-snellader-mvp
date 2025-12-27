import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

TRAIN_IN = "data/mvp_train_enriched_500m.csv"      # jouw beste feature basis (population + OSM 500m)
COMP_IN  = "data/fast_chargers_be_osm.csv"
OUT      = "data/mvp_train_enriched_500m_comp.csv"

RADIUS_M = 500  # concurrentie binnen 500m

def main():
    df = pd.read_csv(TRAIN_IN)
    comp = pd.read_csv(COMP_IN)

    # BallTree verwacht radians (lat, lon)
    df_coords = np.deg2rad(df[["evse_latitude", "evse_longitude"]].values.astype(float))
    comp_coords = np.deg2rad(comp[["lat", "lon"]].values.astype(float))

    tree = BallTree(comp_coords, metric="haversine")

    # radius in radians
    radius_rad = RADIUS_M / 6371000.0

    # tel concurrenten per locatie
    counts = tree.query_radius(df_coords, r=radius_rad, count_only=True)

    df["competitors_fast_500m"] = counts.astype(int)

    print("Voorbeeld (5 rijen):")
    print(df[[
        "evse_latitude",
        "evse_longitude",
        "competitors_fast_500m",
        "sessions_per_day"
    ]].head())

    print("\nVerdeling competitors_fast_500m:")
    print(df["competitors_fast_500m"].describe())

    df.to_csv(OUT, index=False)
    print(f"\n✅ Opgeslagen: {OUT}")

if __name__ == "__main__":
    main()