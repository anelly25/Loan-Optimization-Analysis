# database.py

import pandas as pd
from sqlalchemy import create_engine

def get_database_engine():
    engine = create_engine("mssql+pyodbc://@ANELLY25\\MSSQLSERVER01/TexasLoanStrategyV2?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")
    return engine

def load_data(engine):
    df_loan = pd.read_sql("SELECT * FROM LoanPerformanceData", engine)
    df_macro = pd.read_sql("SELECT * FROM MacroEconomicData", engine)
    # Convert Date columns to datetime
    df_loan["Date"] = pd.to_datetime(df_loan["Date"])
    df_macro["Date"] = pd.to_datetime(df_macro["Date"])
    return df_loan, df_macro

def load_sector_data(engine, table_name):
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    if df.empty:
        print(f"⚠️ WARNING: {table_name} is empty. Skipping merge.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    return df
