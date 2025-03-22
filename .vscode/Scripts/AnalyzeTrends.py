import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# -------------------------------
# 1. Connect to SQL Server and Load Data
# -------------------------------
engine = create_engine("mssql+pyodbc://@ANELLY25\\MSSQLSERVER01/TexasLoanStrategyV2?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

# Load main tables
df_loan = pd.read_sql("SELECT * FROM LoanPerformanceData", engine)
df_macro = pd.read_sql("SELECT * FROM MacroEconomicData", engine)

# Load sector-specific tables
df_ci = pd.read_sql("SELECT * FROM CILoanData", engine)
df_cre = pd.read_sql("SELECT * FROM CRELoanData", engine)
df_mortgage = pd.read_sql("SELECT * FROM MortgageLoanData", engine)
df_construction = pd.read_sql("SELECT * FROM ConstructionLoanData", engine)

# Rename primary key column in LoanPerformanceData to match the foreign keys in sector tables
df_loan.rename(columns={"ID": "LoanPerformanceID"}, inplace=True)

# Ensure all Date columns are in datetime format
df_loan['Date'] = pd.to_datetime(df_loan['Date'])
df_macro['Date'] = pd.to_datetime(df_macro['Date'])
df_ci['Date'] = pd.to_datetime(df_ci['Date'])
df_cre['Date'] = pd.to_datetime(df_cre['Date'])
df_mortgage['Date'] = pd.to_datetime(df_mortgage['Date'])
df_construction['Date'] = pd.to_datetime(df_construction['Date'])

# Adjust MacroEconomicData Date to month-end so it aligns with LoanPerformanceData
df_macro['Date'] = df_macro['Date'].dt.to_period('M').dt.to_timestamp('M')

# -------------------------------
# 2. Merge Data for Analysis
# -------------------------------
# Merge LoanPerformanceData with MacroEconomicData on Date
df_merged = pd.merge(df_loan, df_macro, on="Date", how="left")

# Merge each sector-specific table on LoanPerformanceID and Date
df_merged = pd.merge(df_merged, df_ci, on=["LoanPerformanceID", "Date"], how="left", suffixes=("", "_CI"))
df_merged = pd.merge(df_merged, df_cre, on=["LoanPerformanceID", "Date"], how="left", suffixes=("", "_CRE"))
df_merged = pd.merge(df_merged, df_mortgage, on=["LoanPerformanceID", "Date"], how="left", suffixes=("", "_Mortgage"))
df_merged = pd.merge(df_merged, df_construction, on=["LoanPerformanceID", "Date"], how="left", suffixes=("", "_Construction"))

# Remove duplicate rows if any
df_merged.drop_duplicates(inplace=True)

# Extract Year for aggregation
df_merged['Year'] = df_merged['Date'].dt.year

# Export the cleaned, merged dataset to CSV for Power BI import
df_merged.to_csv("CleanedLoanData.csv", index=False)
print("✅ Cleaned merged dataset exported as CleanedLoanData.csv")

# -------------------------------
# 3. Trend Analysis for Power BI (Including Macroeconomic Factors)
# -------------------------------
# Aggregate DefaultRate, Profitability, and Macroeconomic Factors by Year and LoanCategory
trend_agg = df_merged.groupby(['Year', 'LoanCategory']).agg({
    'DefaultRate': 'mean',
    'Profitability': 'mean',
    'InflationRate': 'mean',
    'GDPGrowthRate': 'mean',
    'UnemploymentRate': 'mean',
    'HomePriceIndex': 'mean'
}).reset_index()

# Export the aggregated trend data to a CSV for Power BI
trend_agg.to_csv("LoanAnalysisData.csv", index=False)
print("✅ Loan analysis data (including macroeconomic factors) exported as LoanAnalysisData.csv")

# -------------------------------
# 4. Compute Correlations for Heatmap
# -------------------------------
# Define loan metrics and macro indicators
loan_metrics = ['DefaultRate', 'Profitability']
macro_indicators = ['InflationRate', 'UnemploymentRate', 'GDPGrowthRate', 'HomePriceIndex']

# Store correlation results
correlation_results = []

for category in df_merged['LoanCategory'].unique():
    df_cat = df_merged[df_merged['LoanCategory'] == category].copy()
    df_cat.dropna(subset=loan_metrics + macro_indicators, inplace=True)

    if df_cat.empty:
        print(f"[Correlation Warning] No complete data available for {category}")
        continue

    # Compute correlations
    for loan_metric in loan_metrics:
        for macro_factor in macro_indicators:
            correlation = df_cat[loan_metric].corr(df_cat[macro_factor])
            correlation_results.append({
                'LoanCategory': category,
                'LoanMetric': loan_metric,
                'MacroFactor': macro_factor,
                'Correlation': correlation
            })

# Convert to DataFrame and Save
df_correlation = pd.DataFrame(correlation_results)
df_correlation.to_csv("LoanMacroCorrelation.csv", index=False)
print("✅ Correlation matrix exported as LoanMacroCorrelation.csv")

# -------------------------------
# 5. Trend Analysis for Default Rate & Profitability Over Time
# -------------------------------
# Plot trends for Default Rate and Profitability by Loan Category using the aggregated data
loan_categories_trend = trend_agg['LoanCategory'].unique()

# Plot Default Rate Trend
plt.figure(figsize=(10, 6))
for category in loan_categories_trend:
    df_cat_trend = trend_agg[trend_agg['LoanCategory'] == category]
    plt.plot(df_cat_trend['Year'], df_cat_trend['DefaultRate'], marker='o', label=category)
plt.title('Default Rate Trend Over Time by Loan Category')
plt.xlabel('Year')
plt.ylabel('Average Default Rate')
plt.legend(title='Loan Category')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Profitability Trend
plt.figure(figsize=(10, 6))
for category in loan_categories_trend:
    df_cat_trend = trend_agg[trend_agg['LoanCategory'] == category]
    plt.plot(df_cat_trend['Year'], df_cat_trend['Profitability'], marker='o', label=category)
plt.title('Profitability Trend Over Time by Loan Category')
plt.xlabel('Year')
plt.ylabel('Average Profitability')
plt.legend(title='Loan Category')
plt.grid(True)
plt.tight_layout()
plt.show()
