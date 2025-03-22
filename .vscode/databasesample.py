import pandas as pd
from sqlalchemy import create_engine

# ✅ Connect to SQL Server
engine = create_engine("mssql+pyodbc://@ANELLY25\\MSSQLSERVER01/TexasLoanStrategyV2?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

# ✅ Load all data
query_loan = "SELECT * FROM LoanPerformanceData"
query_macro = "SELECT * FROM MacroEconomicData"

df_loan = pd.read_sql(query_loan, engine)
df_macro = pd.read_sql(query_macro, engine)

# ✅ Save data as CSV files
df_loan.to_csv("LoanPerformanceData.csv", index=False)
df_macro.to_csv("MacroEconomicData.csv", index=False)

print("✅ Data saved as CSV files: LoanPerformanceData.csv & MacroEconomicData.csv")

