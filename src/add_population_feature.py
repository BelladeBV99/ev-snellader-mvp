import pandas as pd
from pyproj import Transformer

# Load MVP train data
mvp = pd.read_csv("data/mvp_train.csv")

# Load Statbel grid population (1km)
grid = pd.read_csv("data/statbel_population_grid_1km_2020.csv")
pop_map = dict(zip(grid["GRD_NEWID"], grid["MS_POPULATION_20200101"]))

# Transformer: WGS84 -> ETRS89 / LAEA Europe (used by this grid IDs)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

def latlon_to_grid_id_floor(lat, lon):
    x, y = transformer.transform(lon, lat)  # meters
    e_km = int(x // 1000)
    n_km = int(y // 1000)
    return f"1kmN{n_km}E{e_km}"

# Add population feature
grid_ids = []
pops = []

for lat, lon in zip(mvp["evse_latitude"].values, mvp["evse_longitude"].values):
    gid = latlon_to_grid_id_floor(lat, lon)
    grid_ids.append(gid)
    pops.append(int(pop_map.get(gid, 0)))  # if not found -> 0

mvp["grid_id_1km"] = grid_ids
mvp["population_1km"] = pops

print("Voorbeeld (5 rijen):")
print(mvp[["evse_latitude","evse_longitude","grid_id_1km","population_1km","sessions_per_day"]].head())

# Save enriched dataset
mvp.to_csv("data/mvp_train_enriched.csv", index=False)
print("\n✅ Opgeslagen: data/mvp_train_enriched.csv")