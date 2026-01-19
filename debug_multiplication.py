
import pandas as pd

DATA_PATH = "data/base_2014_ec_2020_pop_level2.parquet"
AREA_COL = "area"

def investigate():
    df = pd.read_parquet(DATA_PATH)
    
    # Check total rows
    print(f"Total rows: {len(df)}")
    
    # Check keys
    print(f"Unique Areas: {df[AREA_COL].nunique()}")
    print(f"Unique Times: {df['@time'].unique()}") if '@time' in df.columns else print("No @time column")
    print(f"Unique SIC Codes: {df['sicCode'].nunique()}")
    
    # Check duplicates for a specific area (e.g., 13101 Chiyoda-ku)
    sample = df[df[AREA_COL] == "13101"]
    if not sample.empty:
        print("\n--- Sample (13101) ---")
        print(sample[['@time', 'sicCode', 'population', 'establishments']].head(20))
        print(f"Total rows for 13101: {len(sample)}")
        print(f"Total Population Sum for 13101: {pd.to_numeric(sample['population'], errors='coerce').sum()}")
        print(f"Unique Populations: {sample['population'].unique()}")

if __name__ == "__main__":
    investigate()
