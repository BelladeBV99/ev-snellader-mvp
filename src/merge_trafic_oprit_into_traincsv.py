import pandas as pd

EXCEL_PATH = "data/Verkoopdatagireve.xls"
TRAINCSV_PATH = "data/mvp_train_enriched_500m_comp.csv"
OUT_PATH = "data/mvp_train_enriched_500m_comp.csv"  # overwrite same file

def main():
    # 1) Load current training CSV (has all enrich features)
    train = pd.read_csv(TRAINCSV_PATH)

    # 2) Load excel (has new columns)
    xls = pd.read_excel(EXCEL_PATH)

    # Drop unnamed
    xls = xls.loc[:, ~xls.columns.astype(str).str.startswith("Unnamed:")]

    # Ensure columns exist
    for c in ["evse_latitude", "evse_longitude", "Trafic", "Oprit"]:
        if c not in xls.columns:
            raise ValueError(f"Excel mist kolom: {c}")

    # 3) Keep only mapping columns and clean
    xls_map = xls[["evse_latitude", "evse_longitude", "Trafic", "Oprit"]].copy()
    xls_map["evse_latitude"] = pd.to_numeric(xls_map["evse_latitude"], errors="coerce")
    xls_map["evse_longitude"] = pd.to_numeric(xls_map["evse_longitude"], errors="coerce")
    xls_map["Trafic"] = xls_map["Trafic"].fillna("Unknown").astype(str)
    xls_map["Oprit"] = xls_map["Oprit"].fillna("Unknown").astype(str)

    # 4) Merge on lat/lon (rounded to avoid float mismatch)
    train["_lat5"] = train["evse_latitude"].round(5)
    train["_lon5"] = train["evse_longitude"].round(5)
    xls_map["_lat5"] = xls_map["evse_latitude"].round(5)
    xls_map["_lon5"] = xls_map["evse_longitude"].round(5)

    merged = train.merge(
        xls_map[["_lat5", "_lon5", "Trafic", "Oprit"]],
        on=["_lat5", "_lon5"],
        how="left"
    )

    # 5) Fill missing (if any)
    merged["Trafic"] = merged["Trafic"].fillna("Unknown").astype(str)
    merged["Oprit"] = merged["Oprit"].fillna("Unknown").astype(str)

    # cleanup
    merged = merged.drop(columns=["_lat5", "_lon5"])

    merged.to_csv(OUT_PATH, index=False)

    print("✅ Updated:", OUT_PATH)
    print("Columns now include Trafic/Oprit?", "Trafic" in merged.columns, "Oprit" in merged.columns)
    print("Missing Trafic:", int((merged["Trafic"] == "Unknown").sum()))
    print("Missing Oprit:", int((merged["Oprit"] == "Unknown").sum()))

if __name__ == "__main__":
    main()