import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, roc_curve
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV

# ------------------ Database Connection ------------------
engine = create_engine("mssql+pyodbc://@ANELLY25\\MSSQLSERVER01/TexasLoanStrategyV2?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

# ------------------ Load Data ------------------
df_loan = pd.read_sql("SELECT * FROM LoanPerformanceData", engine)
df_macro = pd.read_sql("SELECT * FROM MacroEconomicData", engine)

# Convert Date Columns
df_loan["Date"] = pd.to_datetime(df_loan["Date"])
df_macro["Date"] = pd.to_datetime(df_macro["Date"])

# ------------------ Load Sector-Specific Data ------------------
def load_sector_data(table_name):
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    if df.empty:
        print(f"⚠️ WARNING: {table_name} is empty. Skipping merge.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is datetime
    return df

df_ci = load_sector_data("CILoanData")
df_cre = load_sector_data("CRELoanData")
df_mortgage = load_sector_data("MortgageLoanData")
df_construction = load_sector_data("ConstructionLoanData")

# ------------------ Merge Macro-Economic Data ------------------
df_macro.set_index("Date", inplace=True)
df_macro = df_macro.resample("ME").ffill().reset_index()
df_combined = pd.merge(df_loan, df_macro, on="Date", how="left")

# ------------------ Merge Sector-Specific Data Using Date ------------------
sector_data = {
    "CILoanData": df_ci, 
    "CRELoanData": df_cre, 
    "MortgageLoanData": df_mortgage, 
    "ConstructionLoanData": df_construction
}

for sector_name, sector_df in sector_data.items():
    if not sector_df.empty and "Date" in sector_df.columns:
        print(f"🔹 Merging {sector_name} into LoanPerformanceData based on Date...")
        df_combined = pd.merge(df_combined, sector_df, on="Date", how="left", suffixes=("", f"_{sector_name}"))

# ------------------ Handle Missing Values ------------------
df_combined.ffill(inplace=True)
df_combined.bfill(inplace=True)

print(f"\n✅ Merged Data Loaded: {df_combined.shape[0]} rows")

# ------------------ Preprocessing ------------------
le = LabelEncoder()

# Encode Loan Categories Before Filtering
df_combined["LoanCategory"] = le.fit_transform(df_combined["LoanCategory"])
loan_category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))  # Save encoding map

# ------------------ Train Separate Models for Each Loan Type ------------------
loan_categories = {
    "C&I": ["LoanInterestRate", "LoanVolume", "InflationRate", "UnemploymentRate", "GDPGrowthRate", 
            "BusinessConfidenceIndex", "IndustrialProduction", "EnergySectorPerformance"],
    "CRE": ["LoanInterestRate", "LoanVolume", "InflationRate", "UnemploymentRate", "GDPGrowthRate", 
            "PropertyPrices", "VacancyRates", "RentGrowthRates"],
    "Mortgage": ["LoanInterestRate", "LoanVolume", "InflationRate", "UnemploymentRate", "GDPGrowthRate", 
                 "HomePriceIndex", "ForeclosureRates", "HousingInventory"],
    "Construction": ["LoanInterestRate", "LoanVolume", "InflationRate", "UnemploymentRate", "GDPGrowthRate", 
                     "ConstructionSpending", "BuildingPermitsIssued", "CommodityPrices"]
}

trained_models = {}
optimal_thresholds = {}

for loan_type, features in loan_categories.items():
    available_features = [f for f in features if f in df_combined.columns]
    
    if not available_features:
        print(f"⚠️ WARNING: No available data for {loan_type}. Skipping training.")
        continue

    # Ensure loan type exists in dataset
    if loan_type not in loan_category_mapping:
        print(f"⚠️ WARNING: Loan type '{loan_type}' not found in dataset. Skipping training.")
        continue

    # Filter data for specific loan category
    df_filtered = df_combined[df_combined["LoanCategory"] == loan_category_mapping[loan_type]]

    if df_filtered.empty:
        print(f"⚠️ WARNING: No data available for {loan_type}. Skipping training.")
        continue

    X = df_filtered[available_features]
    y_profit = df_filtered["Profitability"]
    y_default = (df_filtered["DefaultRate"] > 2.5).astype(int)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Ensure at least 2 samples per class before stratification
    if len(np.unique(y_default)) > 1:
        stratify_label = y_default
    else:
        stratify_label = None  # Avoid stratify error

    try:
        X_train, X_test, y_train_profit, y_test_profit = train_test_split(
            X_scaled, y_profit, test_size=0.2, random_state=42, stratify=stratify_label if stratify_label is not None and len(y_default) > 1 else None
        )
        X_train_class, X_test_class, y_train_default, y_test_default = train_test_split(
            X_scaled, y_default, test_size=0.2, random_state=42, stratify=stratify_label if stratify_label is not None and len(y_default) > 1 else None
        )
    except ValueError:
        print(f"⚠️ WARNING: Skipping classification training for {loan_type} due to insufficient data.")
        continue

    # Train Regression Model
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_profit)

    # Train Logistic Regression Model
    class_model = None
    best_threshold = 0.5  # Default 50% threshold

    if len(np.unique(y_train_default)) > 1:
        class_model = LogisticRegression()
        class_model.fit(X_train_class, y_train_default)

        # Find optimal threshold using ROC curve
        y_probs = class_model.predict_proba(X_test_class)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_default, y_probs)

        best_threshold = thresholds[np.argmax(tpr - fpr)]  # Maximize TPR - FPR
        optimal_thresholds[loan_type] = best_threshold

        print(f"\n📊 Optimal Probability Threshold for {loan_type}: {best_threshold:.2f}")
        print(f"✅ Logistic Regression Accuracy for {loan_type}: {accuracy_score(y_test_default, (y_probs >= best_threshold).astype(int)):.4f}")

    trained_models[loan_type] = {
        "regressor": reg_model,
        "classifier": class_model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": available_features,
        "threshold": best_threshold,
        "X_train_class": X_train_class,   # ✅ Add this
    "X_test_class": X_test_class,     # ✅ Add this
    "y_train_default": y_train_default,  # ✅ Add this
    "y_test_default": y_test_default    # ✅ Add this
    }

# ------------------ Loan Allocation Strategy ------------------
def get_optimal_loan_mix(loan_type):
    if loan_type not in trained_models:
        print(f"⚠️ WARNING: No trained model available for {loan_type}. Skipping prediction.")
        return

    model_data = trained_models[loan_type]
    features = model_data["feature_names"]

    economic_df = pd.DataFrame([[3.5, 1000000, 2.5, 4.0, 2.0] + [250000]*len(features[5:])], columns=features)
    economic_imputed = model_data["imputer"].transform(economic_df)
    economic_scaled = model_data["scaler"].transform(economic_imputed)

    profit_pred = model_data["regressor"].predict(economic_scaled)
    default_prob = model_data["classifier"].predict_proba(economic_scaled)[:, 1] if model_data["classifier"] else "N/A"

    print(f"\n📊 Recommended Loan Allocation for {loan_type}:")
    print(f"Predicted Profitability: {profit_pred[0]:.2f}, Adjusted Default Risk: {default_prob if default_prob == 'N/A' else f'{default_prob[0]:.2%}'}")

# Run Predictions
for loan in trained_models.keys():
    get_optimal_loan_mix(loan)


# ------------------ Train & Optimize XGBoost for Default Classification ------------------
for loan_type, model_data in trained_models.items():
    print(f"\n🔹 Training & Tuning XGBoost for {loan_type}...")

    # Retrieve training and testing data
    X_train_class, X_test_class = model_data["X_train_class"], model_data["X_test_class"]
    y_train_default, y_test_default = model_data["y_train_default"], model_data["y_test_default"]

    # Define XGBoost model
    xgb_model = XGBClassifier(eval_metric="logloss")

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_class, y_train_default)

    # Apply the best found parameters
    best_xgb_model = grid_search.best_estimator_

    print(f"\n✅ Best XGBoost Parameters for {loan_type}: {grid_search.best_params_}")
    print(f"✅ Best XGBoost Accuracy for {loan_type}: {grid_search.best_score_:.4f}")

    # Store the trained model
    trained_models[loan_type]["xgboost_classifier"] = best_xgb_model

# ------------------ Loan Allocation Strategy with Optimized XGBoost ------------------
for loan in trained_models.keys():
    print(f"\n📊 Recommended Loan Allocation for {loan} (Optimized XGBoost):")

    # Prepare sample input
    economic_df = pd.DataFrame([[3.5, 1000000, 2.5, 4.0, 2.0] + [250000] * (len(trained_models[loan]["feature_names"]) - 5)],
                                columns=trained_models[loan]["feature_names"])
    economic_imputed = trained_models[loan]["imputer"].transform(economic_df)
    economic_scaled = trained_models[loan]["scaler"].transform(economic_imputed)

    # Default Risk Prediction using Optimized XGBoost
    xgb_optimized_prob = trained_models[loan]["xgboost_classifier"].predict_proba(economic_scaled)[:, 1]

    print(f"Predicted Profitability: N/A (Focusing on Default Risk), Adjusted Default Risk (Optimized XGBoost): {xgb_optimized_prob[0]:.2%}")

def tune_xgboost_with_gridsearch(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = XGBClassifier(eval_metric="logloss")
    
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best XGBoost Parameters: {grid_search.best_params_}")
    print(f"✅ Best XGBoost Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# ------------------ Apply GridSearch to Each Loan Type ------------------
for loan_type, model_data in trained_models.items():
    print(f"\n🔹 Tuning XGBoost for {loan_type} with GridSearchCV...")

    # Retrieve training data
    X_train_class, y_train_default = model_data["X_train_class"], model_data["y_train_default"]

    # Tune XGBoost model
    best_xgb_model = tune_xgboost_with_gridsearch(X_train_class, y_train_default)

    # Store the best model
    trained_models[loan_type]["xgboost_classifier"] = best_xgb_model