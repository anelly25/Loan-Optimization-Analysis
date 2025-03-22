from database import get_database_engine, load_data, load_sector_data
from preprocessing import merge_macro_data, preprocess_data, merge_sector_data, handle_missing_values
from modeling import encode_loan_categories, train_models, tune_xgboost_models
from utils import get_optimal_loan_mix, save_predictions_to_csv

def main():
    # ------------------ Database Connection and Data Loading ------------------
    engine = get_database_engine()
    df_loan, df_macro = load_data(engine)
    
    # Load sector-specific data
    df_ci = load_sector_data(engine, "CILoanData")
    df_cre = load_sector_data(engine, "CRELoanData")
    df_mortgage = load_sector_data(engine, "MortgageLoanData")
    df_construction = load_sector_data(engine, "ConstructionLoanData")
    
    # ------------------ Preprocessing ------------------
    df_macro = merge_macro_data(df_macro)
    df_combined = preprocess_data(df_loan, df_macro)
    sector_data = {
        "CILoanData": df_ci,
        "CRELoanData": df_cre,
        "MortgageLoanData": df_mortgage,
        "ConstructionLoanData": df_construction
    }
    df_combined = merge_sector_data(df_combined, sector_data)
    df_combined = handle_missing_values(df_combined)
    print(f"Merged Data Loaded: {df_combined.shape[0]} rows")
    
    # Encode Loan Categories
    df_combined, loan_category_mapping = encode_loan_categories(df_combined)
    
    # ------------------ Model Training ------------------
    trained_models = train_models(df_combined, loan_category_mapping)
    
    # ------------------ Define Multiple Economic Scenarios ------------------
    economic_scenarios = [
        {"ScenarioName": "Baseline", "InflationRate": 2.5, "GDPGrowthRate": 2.0, "UnemploymentRate": 4.0},
        {"ScenarioName": "High Inflation", "InflationRate": 7.0, "GDPGrowthRate": 1.5, "UnemploymentRate": 5.5},
        {"ScenarioName": "Recession", "InflationRate": 3.5, "GDPGrowthRate": -1.0, "UnemploymentRate": 6.5},
        {"ScenarioName": "Boom Economy", "InflationRate": 1.5, "GDPGrowthRate": 4.0, "UnemploymentRate": 3.0},
        {"ScenarioName": "Stagflation", "InflationRate": 8.0, "GDPGrowthRate": -2.0, "UnemploymentRate": 7.0}
    ]
    
    # ------------------ Loan Allocation Strategy (Standard Models) ------------------
    predictions = []
    for loan in trained_models.keys():
        pred = get_optimal_loan_mix(loan, trained_models, economic_scenarios, optimized=False)
        predictions.extend(pred)  # Add all scenario predictions
    
    # ------------------ XGBoost Tuning ------------------
    tuned_models = tune_xgboost_models(trained_models)
    
    # ------------------ Loan Allocation Strategy with Optimized XGBoost ------------------
    for loan in tuned_models.keys():
        pred = get_optimal_loan_mix(loan, tuned_models, economic_scenarios, optimized=True)
        predictions.extend(pred)
    
    # ------------------ Save Predictions to CSV ------------------
    save_predictions_to_csv(predictions)

if __name__ == '__main__':
    main()
