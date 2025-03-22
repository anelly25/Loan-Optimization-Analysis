# 💼 Loan Optimization & Performance Analysis

This project analyzes and predicts the **profitability** and **default risk** of various loan types under different macroeconomic conditions using historical data.

We also optimize the **loan portfolio mix** to balance return and risk across multiple economic scenarios — visualized with Power BI.

---

## 🔍 Project Goals

- Understand how inflation, GDP, and unemployment impact loan performance.
- Train models to predict loan profitability and default risk by sector.
- Simulate scenarios (like recession, stagflation, boom economy).
- Recommend optimal loan allocations that maximize profit while minimizing risk.

---

## 📊 Tools & Technologies

| Area | Tools |
|------|-------|
| Language | Python |
| ML Models | Random Forest, Logistic Regression, XGBoost |
| Data | SQL Server (imported to Pandas) |
| Optimization | `scipy.optimize.linprog` |
| Visualization | Power BI |
| Repo Management | Git & GitHub |

---

## 🧠 How It Works

1. **Data Collection**:
   - Loan performance data (`LoanPerformanceData`)
   - Macroeconomic indicators (`MacroEconomicData`)
   - Sector-specific loan data (C&I, CRE, Mortgage, Construction)

2. **Preprocessing**:
   - Merge datasets on `Date`
   - Forward-fill missing values
   - Encode `LoanCategory`

3. **Modeling**:
   - Train:
     - `RandomForestRegressor` → Profitability
     - `LogisticRegression` & `XGBoostClassifier` → Default Risk
   - Predict based on current & future macroeconomic inputs

4. **Optimization**:
   - Use linear programming to **recommend allocation percentages** to loan types
   - Objective: Maximize total predicted profitability
   - Constraint: Stay below a given risk tolerance (e.g. 50%)

5. **Visualization (Power BI)**:
   - 📈 Page 1: Predicted loan performance by scenario
   - 📉 Page 2: Trends + macro correlations
   - 📐 Page 3: Optimal loan mix (based on scenario)

---

## 📁 Files in This Repo

| File | Purpose |
|------|---------|
| `main.py` | Main training pipeline |
| `database.py` | SQL connection + data loading |
| `modeling.py` | Model training and tuning |
| `utils.py` | Optimization functions & helpers |
| `preprocessing.py` | Data cleaning and merging |
| `LoanAnalysisData.csv` | Aggregated trends for Power BI |
| `OptimalLoanAllocation.csv` | Optimal loan mix by scenario |
| `LoanMacroCorrelation.csv` | Correlation data for heatmaps |
| `loan_predictions.csv` | Profitability + default risk results |
| `PowerBI Report` | Visual insights from the model |

---

## 📌 What I’d Improve Next

- Build a **dashboard with Streamlit** or Flask so users can adjust scenarios live
- Add **loan-level granularity** (e.g. credit score, term, LTV)
- Add **scenario-based stress testing** (monte carlo simulations)
- Incorporate **external data sources** (FRED API, Census, BLS)

---

## 📣 About Me

Hi, I'm **Ashton Nelson** — a data analyst and software developer passionate about using data and AI to solve real-world business challenges. I'm currently exploring roles in finance, analytics, and machine learning.

📧 Email: [ashtonnelson28@gmail.com](mailto:ashtonnelson28@gmail.com)  
📈 LinkedIn: [linkedin.com/in/your-link](https://linkedin.com) (replace with yours!)

---

## ⭐ Give this repo a star if you found it interesting or helpful!
