import pandas as pd

# Load data
df = pd.read_parquet("data/base_2014_ec_2020_pop_level2.parquet")
df["area"] = df["area"].astype(str).str.zfill(5)

# Check Kanagawa (14) for Yokohama (14100) and its wards (14101...)
kanagawa = df[df["area"].str.startswith("14")].drop_duplicates("area")
print("Kanagawa Areas:")
print(kanagawa[kanagawa["area"].str.startswith("141")][["area", "areaName"]])

# Check if 14100 exists
has_yokohama_total = "14100" in kanagawa["area"].values
has_yokohama_ward = "14101" in kanagawa["area"].values

print(f"\nHas Yokohama Total (14100): {has_yokohama_total}")
print(f"Has Yokohama Ward (14101): {has_yokohama_ward}")

if has_yokohama_total and has_yokohama_ward:
    print("DUPLICATE DETECTED: Both City Total and Wards are present.")
