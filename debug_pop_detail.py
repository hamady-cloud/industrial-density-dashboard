
import pandas as pd

DATA_PATH = "data/base_2014_ec_2020_pop_level2.parquet"
AREA_COL = "area"

def analyze_population():
    df = pd.read_parquet(DATA_PATH)
    
    # Filter 00000
    df = df[df[AREA_COL] != "00000"]
    # Filter XX000 (Pref)
    df = df[~df[AREA_COL].str.endswith("000")]
    
    # Get Max population per area (to solve industry duplication issue locally for check)
    unique_areas = df.groupby(AREA_COL, as_index=False)["population"].max()
    
    total_pop = unique_areas["population"].sum()
    print(f"Total Population (minus 00000 and XX000): {total_pop:,.0f}")
    
    # Check for Designated Cities (suffix 00)
    city_totals = unique_areas[unique_areas[AREA_COL].str.endswith("00")]
    print(f"City Totals (ends with 00, not 000): {len(city_totals)} areas")
    print(f"Sum of City Totals: {city_totals['population'].sum():,.0f}")
    
    # Check Tokyo 23 wards special case? coverage?
    # Tokyo Check
    tokyo = unique_areas[unique_areas[AREA_COL].str.startswith("13")]
    print(f"Tokyo Total: {tokyo['population'].sum():,.0f}")
    
    # Print discrepancies
    # We expect duplicates if City Total + Wards exist
    # Let's verify overlapping
    
    duplicates = []
    
    for code in city_totals[AREA_COL]:
        prefix = code[:3]
        # Check if there are wards (same prefix, not same code)
        wards = unique_areas[(unique_areas[AREA_COL].str.startswith(prefix)) & (unique_areas[AREA_COL] != code)]
        if not wards.empty:
            duplicates.append({
                "city": code,
                "city_pop": float(unique_areas[unique_areas[AREA_COL] == code]["population"].values[0]),
                "wards_cnt": len(wards),
                "wards_pop_sum": wards["population"].sum()
            })
            
    dup_df = pd.DataFrame(duplicates)
    if not dup_df.empty:
        print(f"\nPotential Duplicates Found: {len(dup_df)}")
        print(f"Sum of City Pop (Duplicates): {dup_df['city_pop'].sum():,.0f}")
        print(f"Sum of Wards Pop (Children): {dup_df['wards_pop_sum'].sum():,.0f}")
        
    final_calc = total_pop - dup_df['city_pop'].sum()
    print(f"\nEstimated Correct Pop (Total - CityTotals): {final_calc:,.0f}")

if __name__ == "__main__":
    analyze_population()
