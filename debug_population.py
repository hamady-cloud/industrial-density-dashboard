
import pandas as pd

DATA_PATH = "data/base_2014_ec_2020_pop_level2.parquet"
AREA_COL = "area"

def check_duplicates():
    df = pd.read_parquet(DATA_PATH)
    df[AREA_COL] = df[AREA_COL].astype(str).str.zfill(5)
    
    # Check Kanagawa (14) as an example
    kanagawa = df[df[AREA_COL].str.startswith("14")].copy()
    kanagawa = kanagawa.drop_duplicates(subset=[AREA_COL])
    
    print("--- Kanagawa Records ---")
    print(kanagawa[[AREA_COL, "areaName", "population"]].sort_values(AREA_COL).head(20))
    
    # Check if "14100" (Yokohama-shi) exists along with "14101" (Tsurumi-ku)
    yokohama_shi = kanagawa[kanagawa[AREA_COL] == "14100"]
    yokohama_wards = kanagawa[kanagawa[AREA_COL].str.startswith("141") & (kanagawa[AREA_COL] != "14100")]
    
    print("\n--- Yokohama Check ---")
    print(f"Yokohama City Total (14100) Exists: {not yokohama_shi.empty}")
    if not yokohama_shi.empty:
        print(f"Population: {yokohama_shi['population'].values[0]}")
        
    print(f"Yokohama Wards Count: {len(yokohama_wards)}")
    print(f"Wards Population Sum: {yokohama_wards['population'].sum()}")

if __name__ == "__main__":
    check_duplicates()
