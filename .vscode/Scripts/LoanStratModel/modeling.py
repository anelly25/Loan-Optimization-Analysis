# modeling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from xgboost import XGBClassifier

def encode_loan_categories(df):
    le = LabelEncoder()
    df["LoanCategory"] = le.fit_transform(df["LoanCategory"])
    loan_category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return df, loan_category_mapping

def train_models(df_combined, loan_category_mapping):
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
    
    for loan_type, features in loan_categories.items():
        available_features = [f for f in features if f in df_combined.columns]
        if not available_features:
            print(f"⚠️ WARNING: No available data for {loan_type}. Skipping training.")
            continue
        if loan_type not in loan_category_mapping:
            print(f"⚠️ WARNING: Loan type '{loan_type}' not found in dataset. Skipping training.")
            continue
        
        df_filtered = df_combined[df_combined["LoanCategory"] == loan_category_mapping[loan_type]]
        if df_filtered.empty:
            print(f"⚠️ WARNING: No data available for {loan_type}. Skipping training.")
            continue
        
        X = df_filtered[available_features]
        y_profit = df_filtered["Profitability"]
        y_default = (df_filtered["DefaultRate"] > 2.5).astype(int)
        
        # Impute missing values and scale features
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        stratify_label = y_default if len(np.unique(y_default)) > 1 else None
        
        try:
            X_train, X_test, y_train_profit, y_test_profit = train_test_split(
                X_scaled, y_profit, test_size=0.2, random_state=42, stratify=stratify_label
            )
            X_train_class, X_test_class, y_train_default, y_test_default = train_test_split(
                X_scaled, y_default, test_size=0.2, random_state=42, stratify=stratify_label
            )
        except ValueError:
            print(f"⚠️ WARNING: Skipping classification training for {loan_type} due to insufficient data.")
            continue
        
        # Train Regression Model
        reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_model.fit(X_train, y_train_profit)
        
        # Train Classification Model (Logistic Regression)
        class_model = None
        best_threshold = 0.5
        if len(np.unique(y_train_default)) > 1:
            class_model = LogisticRegression()
            class_model.fit(X_train_class, y_train_default)
            
            y_probs = class_model.predict_proba(X_test_class)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test_default, y_probs)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            print(f"\n📊 Optimal Probability Threshold for {loan_type}: {best_threshold:.2f}")
            print(f"✅ Logistic Regression Accuracy for {loan_type}: {accuracy_score(y_test_default, (y_probs >= best_threshold).astype(int)):.4f}")
        
        trained_models[loan_type] = {
            "regressor": reg_model,
            "classifier": class_model,
            "scaler": scaler,
            "imputer": imputer,
            "feature_names": available_features,
            "threshold": best_threshold,
            "X_train_class": X_train_class,
            "X_test_class": X_test_class,
            "y_train_default": y_train_default,
            "y_test_default": y_test_default
        }
    return trained_models

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

def tune_xgboost_models(trained_models):
    for loan_type, model_data in trained_models.items():
        print(f"\n🔹 Tuning XGBoost for {loan_type} with GridSearchCV...")
        X_train_class, y_train_default = model_data["X_train_class"], model_data["y_train_default"]
        best_xgb_model = tune_xgboost_with_gridsearch(X_train_class, y_train_default)
        trained_models[loan_type]["xgboost_classifier"] = best_xgb_model
        
        # Demonstrate prediction using the optimized XGBoost model
        economic_df = pd.DataFrame(
            [[3.5, 1000000, 2.5, 4.0, 2.0] + [250000] * (len(model_data["feature_names"]) - 5)],
            columns=model_data["feature_names"]
        )
        economic_imputed = model_data["imputer"].transform(economic_df)
        economic_scaled = model_data["scaler"].transform(economic_imputed)
        xgb_optimized_prob = best_xgb_model.predict_proba(economic_scaled)[:, 1]
        print(f"\n📊 Recommended Loan Allocation for {loan_type} (Optimized XGBoost):")
        print(f"Predicted Profitability: N/A (Focusing on Default Risk), Adjusted Default Risk (Optimized XGBoost): {xgb_optimized_prob[0]:.2%}")
    return trained_models
