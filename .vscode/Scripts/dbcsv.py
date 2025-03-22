import pandas as pd
from sqlalchemy import create_engine

# Connect to database
engine = create_engine("mssql+pyodbc://@ANELLY25\\MSSQLSERVER01/TexasLoanStrategy?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

# List of tables to export
tables = ["LoanPerformanceData", "MacroEconomicData", "CILoanData", "CRELoanData", "MALoanData", 
          "RetailLoanData", "MortgageLoanData", "ConstructionLoanData", "ConsumerLoanData"]

for table in tables:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    df.to_csv(f"{table}.csv", index=False)
    print(f"✅ Exported {table}.csv")

print("✅ All tables exported successfully.")
