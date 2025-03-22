import pandas as pd

def get_optimal_loan_mix(loan_type, trained_models, economic_scenarios, optimized=False):
    """
    Computes the recommended loan allocation for a given loan type across multiple scenarios.
    Includes macroeconomic variables in the output.
    """
    predictions = []

    for scenario in economic_scenarios:
        model_data = trained_models[loan_type]
        features = model_data["feature_names"]

        # Ensure that the scenario has values for the required features
        scenario_data = [scenario.get(f, 250000) for f in features]
        economic_df = pd.DataFrame([scenario_data], columns=features)

        # Impute and scale the input data
        economic_imputed = model_data["imputer"].transform(economic_df)
        economic_scaled = model_data["scaler"].transform(economic_imputed)

        # Extract macroeconomic variables for reference
        inflation_rate = scenario.get("InflationRate", None)
        gdp_growth_rate = scenario.get("GDPGrowthRate", None)
        unemployment_rate = scenario.get("UnemploymentRate", None)

        if not optimized:
            profit_pred = model_data["regressor"].predict(economic_scaled)[0]
            default_prob = None
            if model_data["classifier"]:
                default_prob = model_data["classifier"].predict_proba(economic_scaled)[:, 1][0]
            
            predictions.append({
                "Scenario": scenario["ScenarioName"],
                "LoanType": loan_type,
                "PredictedProfitability": profit_pred,
                "AdjustedDefaultRisk": default_prob,
                "Method": "Standard",
                "InflationRate": inflation_rate,
                "GDPGrowthRate": gdp_growth_rate,
                "UnemploymentRate": unemployment_rate
            })
        else:
            if "xgboost_classifier" in model_data:
                xgb_prob = model_data["xgboost_classifier"].predict_proba(economic_scaled)[:, 1][0]
                predictions.append({
                    "Scenario": scenario["ScenarioName"],
                    "LoanType": loan_type,
                    "PredictedProfitability": None,
                    "AdjustedDefaultRisk": xgb_prob,
                    "Method": "Optimized XGBoost",
                    "InflationRate": inflation_rate,
                    "GDPGrowthRate": gdp_growth_rate,
                    "UnemploymentRate": unemployment_rate
                })
            else:
                predictions.append({
                    "Scenario": scenario["ScenarioName"],
                    "LoanType": loan_type,
                    "PredictedProfitability": None,
                    "AdjustedDefaultRisk": None,
                    "Method": "Optimized XGBoost - Not Available",
                    "InflationRate": inflation_rate,
                    "GDPGrowthRate": gdp_growth_rate,
                    "UnemploymentRate": unemployment_rate
                })

    return predictions


def save_predictions_to_csv(predictions, filename="loan_predictions.csv"):
    """
    Saves the list of prediction dictionaries to a CSV file.
    """
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    print(f"✅ Predictions saved to {filename}")
