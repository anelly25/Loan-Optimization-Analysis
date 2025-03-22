# preprocessing.py

import pandas as pd

def merge_macro_data(df_macro):
    df_macro.set_index("Date", inplace=True)
    df_macro = df_macro.resample("ME").ffill().reset_index()
    return df_macro

def preprocess_data(df_loan, df_macro):
    # Merge Loan Performance and Macro-Economic Data on Date
    df_combined = pd.merge(df_loan, df_macro, on="Date", how="left")
    return df_combined

def merge_sector_data(df_combined, sector_data):
    for sector_name, sector_df in sector_data.items():
        if not sector_df.empty and "Date" in sector_df.columns:
            print(f"🔹 Merging {sector_name} into LoanPerformanceData based on Date...")
            df_combined = pd.merge(df_combined, sector_df, on="Date", how="left", suffixes=("", f"_{sector_name}"))
    return df_combined

def handle_missing_values(df):
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
