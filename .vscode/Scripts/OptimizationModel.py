import pandas as pd
import numpy as np
from scipy.optimize import linprog

# -------------------------------
# 1. Load Scenario-Based Loan Data
# -------------------------------
df_scenario = pd.read_csv("loan_predictions.csv")  # Load predicted profitability & risk data

# Ensure numeric columns are properly formatted
df_scenario["PredictedProfitability"] = pd.to_numeric(df_scenario["PredictedProfitability"], errors="coerce")
df_scenario["AdjustedDefaultRisk"] = pd.to_numeric(df_scenario["AdjustedDefaultRisk"], errors="coerce")

# Drop rows with missing values (if any)
df_scenario.dropna(subset=["PredictedProfitability", "AdjustedDefaultRisk"], inplace=True)

# -------------------------------
# 2. Define Optimization Function
# -------------------------------
def optimize_loan_allocation(df, risk_tolerance=0.5):
    """
    Computes the optimal loan allocation to maximize profitability while managing risk.
    risk_tolerance: Lower values (0-0.5) favor low-risk loans; Higher values (0.5-1) favor high returns.
    """

    # Extract profit and risk data
    profits = df["PredictedProfitability"].values
    risks = df["AdjustedDefaultRisk"].values

    # Objective: Maximize total profit -> linprog minimizes, so we use negative profits
    objective = -profits

    # Constraints:
    # - Total allocation must sum to 1 (100% allocation)
    # - Risk constraint based on tolerance (weighted sum of risk)
    A_eq = [np.ones(len(df))]  # Sum of allocations = 1
    b_eq = [1]  # 100% of capital must be allocated

    A_ub = [risks]  # Risk constraint: sum(risk * allocation) <= risk_tolerance
    b_ub = [risk_tolerance]

    # Bounds: Each loan allocation should be between 0% and 100%
    bounds = [(0, 1)] * len(df)

    # Solve optimization
    result = linprog(c=objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        df = df.copy()  # Avoid modifying the original DataFrame
        df.loc[:, "OptimalAllocation"] = result.x  # Assign allocation safely
        return df
    else:
        raise ValueError("Optimization failed: No valid allocation found.")

# -------------------------------
# 3. Run Optimization for Each Scenario
# -------------------------------
scenarios = df_scenario["Scenario"].unique()  # Get all unique macroeconomic scenarios
optimized_allocations = []

for scenario in scenarios:
    df_filtered = df_scenario[df_scenario["Scenario"] == scenario]  # Filter data for the current scenario
    optimal_alloc = optimize_loan_allocation(df_filtered, risk_tolerance=0.5)  # Adjust risk tolerance as needed
    
    # Create a copy and assign scenario safely
    optimal_alloc = optimal_alloc.copy()
    optimal_alloc.loc[:, "Scenario"] = scenario

    optimized_allocations.append(optimal_alloc)

# -------------------------------
# 4. Save Results to CSV for Power BI
# -------------------------------
df_optimal = pd.concat(optimized_allocations, ignore_index=True)  # Merge all optimized scenarios
df_optimal.to_csv("OptimalLoanAllocation.csv", index=False)

print("✅ Optimal loan allocation saved to OptimalLoanAllocation.csv")
