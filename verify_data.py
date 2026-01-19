import pandas as pd
import sys

# Load data
try:
    df = pd.read_parquet("data/base_2014_ec_2020_pop_level2.parquet")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

df["area"] = df["area"].astype(str).str.zfill(5)
df["sicCode"] = df["sicCode"].astype(str).str.strip()
TOTAL_CODE = "__TOTAL__"

# ---- Logic from app.py (Re-implemented for independent check) ----

# 1. Scope Filter (Remove Nation 00000, Pref totals XX000)
#    AND THE FIX for Duplicates (Designated Cities XX100, XX130, XX150)
d = df[df["area"] != "00000"]
d = d[d["population"] > 0]
d = d[~d["area"].str.endswith("000")] # Remove Prefectures

# Duplicate Removal Logic
ignore_suffixes = ["100", "130", "140", "150"]
d_filtered = d[~d["area"].str.endswith(tuple(ignore_suffixes))].copy()

# 2. Industry Aggregation (Total)
# Group by Area and Sum
d_total_filtered = (
    d_filtered.groupby(["area", "areaName"], as_index=False)
    .agg(
        {
            "establishments": "sum",
            "employees": "sum",
            "population": "max", # Assuming population is duplicated per industry row
        }
    )
)

print("-" * 30)
print("VERIFICATION RESULTS (National Total)")
print("-" * 30)

est_sum = d_total_filtered["establishments"].sum()
emp_sum = d_total_filtered["employees"].sum()
pop_sum = d_total_filtered["population"].sum()

print(f"Total Establishments: {est_sum:,.0f}")
print(f"Total Employees:      {emp_sum:,.0f}")
print(f"Total Population:     {pop_sum:,.0f}")

print("-" * 30)
print("Benchmarks (Approx 2014 Economic Census & 2020 Pop):")
print("Establishments: ~5.5 - 5.8 Million")
print("Employees:      ~55 - 60 Million")
print("Population:     ~126 - 127 Million")
print("-" * 30)

# Check for anomalies
if est_sum > 10_000_000:
    print("WARNING: Establishments significantly high (Possible duplication)")
if emp_sum > 100_000_000:
    print("WARNING: Employees significantly high (Possible duplication)")
if pop_sum > 130_000_000:
    print("WARNING: Population significantly high (Possible duplication)")
