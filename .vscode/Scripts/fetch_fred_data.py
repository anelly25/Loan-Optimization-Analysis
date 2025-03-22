import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine, text

# -- Step 1: Database Connection --
engine = create_engine("mssql+pyodbc://@ANELLY25\\MSSQLSERVER01/TexasLoanStrategyV2?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")
conn = engine.connect()

# -- Step 2: Helper Function for Upsert --
def upsert_data(table_name, data, primary_key):
    """Upserts data into the specified SQL table."""
    if data.empty:
        print(f"[Insert] No data available for {table_name}, skipping insert.")
        return

    try:
        if table_name == "LoanPerformanceData":
            conn.execute(text(f"SET IDENTITY_INSERT {table_name} ON"))

        for _, row in data.iterrows():
            conn.execute(text(f"""
                MERGE INTO {table_name} AS target
                USING (SELECT :{primary_key} AS {primary_key}) AS source
                ON target.{primary_key} = source.{primary_key}
                WHEN MATCHED THEN
                    UPDATE SET {', '.join([f"{col} = :{col}" for col in data.columns if col != primary_key])}
                WHEN NOT MATCHED THEN
                    INSERT ({', '.join(data.columns)})
                    VALUES ({', '.join([f":{col}" for col in data.columns])});
            """), row.to_dict())

        if table_name == "LoanPerformanceData":
            conn.execute(text(f"SET IDENTITY_INSERT {table_name} OFF"))

        conn.commit()
    except Exception as e:
        print(f"[Insert Error] {table_name}: {e}")

# -- Step 3: Fetch Macro-Economic Data --
def fetch_macro_data():
    """Fetches and inserts macroeconomic data from FRED API."""
    FRED_API_KEY = "198e7b8e24fa0baec42d61db69488a70"
    series = {
        "InflationRate": "CPIAUCSL",
        "UnemploymentRate": "TXUR",
        "GDPGrowthRate": "TXRGSP",
        "HomePriceIndex": "TXSTHPI"
    }

    df_merged = None
    for key, series_id in series.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        response = requests.get(url).json()

        if 'observations' not in response:
            print(f"[API Error] No observations for '{key}'.")
            continue

        df_series = pd.DataFrame(response['observations'])
        df_series = df_series[['date', 'value']].rename(columns={'date': 'Date', 'value': key})
        df_series[key] = pd.to_numeric(df_series[key], errors='coerce')

        df_merged = df_series if df_merged is None else df_merged.merge(df_series, on='Date', how='outer')

    if df_merged is not None and not df_merged.empty:
        df_merged.dropna(inplace=True)
    else:
        df_merged = pd.DataFrame()

    upsert_data("MacroEconomicData", df_merged, "Date")

fetch_macro_data()

# -- Step 4: Generate Loan Performance Data --
def generate_loan_performance_data():
    np.random.seed(42)

# 🔹 Instead of querying the database, hardcode the known date range
start_year = 1997
end_year = 2023

num_entries = (end_year - start_year + 1) * 12  # One entry per month in range

data = pd.DataFrame({
    "ID": np.arange(1, num_entries + 1),
    "BankID": np.random.randint(1, 51, num_entries),
    "BankName": [f"Bank {i}" for i in range(1, num_entries + 1)],
    "Date": pd.date_range(start=f"1/1/{start_year}", periods=num_entries, freq="ME"),  # 🔹 Adjusted Start Date
    "LoanCategory": np.random.choice(["C&I", "CRE", "M&A", "Retail", "Mortgage", "Construction"], num_entries),
    "LoanInterestRate": np.random.uniform(2, 10, num_entries),
    "LoanVolume": np.random.uniform(100000, 1000000, num_entries),
    "DefaultRate": np.random.uniform(0.5, 5, num_entries),
    "Profitability": np.random.uniform(0.1, 10, num_entries)
})

upsert_data("LoanPerformanceData", data, "ID")

generate_loan_performance_data()

conn.commit()

# -- Step 5: Retrieve Loan Performance Data --
def get_loan_performance_data():
    df = pd.read_sql("SELECT ID, Date FROM LoanPerformanceData", engine)
    if df.empty:
        print("[Data Warning] LoanPerformanceData is still empty! Check database connection.")
    return df

loan_performance_data = get_loan_performance_data()

if loan_performance_data.empty:
    print("[Data Warning] LoanPerformanceData is empty! Skipping dependent table inserts.")
else:
    valid_ids = loan_performance_data["ID"].tolist()
    valid_dates = loan_performance_data["Date"].tolist()

    # -- Step 6: Generate Corrected Sector-Specific Data (Without BankID) --

    def generate_sector_data(table_name, features):
        sector_data = pd.DataFrame({
            "LoanPerformanceID": np.random.choice(valid_ids, 200, replace=True),
            "Date": np.random.choice(valid_dates, 200, replace=True)
        })

        for feature, (low, high) in features.items():
            sector_data[feature] = np.random.uniform(low, high, 200)

        upsert_data(table_name, sector_data, "LoanPerformanceID")

    # **C&I Sector**
    generate_sector_data("CILoanData", {
        "BusinessConfidenceIndex": (50, 100),
        "IndustrialProduction": (0.5, 5),
        "EnergySectorPerformance": (80, 150)
    })

    # **CRE Sector**
    generate_sector_data("CRELoanData", {
        "PropertyPrices": (100000, 500000),
        "VacancyRates": (2, 12),
        "RentGrowthRates": (1, 6)
    })

    # **Mortgage Sector**
    generate_sector_data("MortgageLoanData", {
        "HomePriceIndex": (150000, 600000),
        "ForeclosureRates": (0.5, 5),
        "HousingInventory": (500, 5000)
    })

    # **Construction Sector**
    generate_sector_data("ConstructionLoanData", {
        "ConstructionSpending": (50000, 500000),
        "BuildingPermitsIssued": (10, 500),
        "CommodityPrices": (50, 300)
    })

# -- Step 7: Close the Connection --
conn.close()
print("✅ Database population complete!")
