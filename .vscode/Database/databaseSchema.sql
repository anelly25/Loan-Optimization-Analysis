-- Loan Performance Data
CREATE TABLE LoanPerformanceData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    BankID INT NOT NULL,
    BankName VARCHAR(255) NOT NULL,
    Date DATE NOT NULL,
    LoanCategory VARCHAR(50) NOT NULL,
    LoanInterestRate FLOAT NOT NULL,
    LoanVolume FLOAT NOT NULL,
    DefaultRate FLOAT NOT NULL,
    Profitability FLOAT NOT NULL
);

-- Macro-Economic Data
CREATE TABLE MacroEconomicData (
    Date DATE PRIMARY KEY,
    InflationRate FLOAT NOT NULL,
    UnemploymentRate FLOAT NOT NULL,
    GDPGrowthRate FLOAT NOT NULL,
    HomePriceIndex FLOAT NOT NULL
);

-- C&I Loan Data
CREATE TABLE CILoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    BusinessConfidenceIndex FLOAT NOT NULL,
    IndustrialProduction FLOAT NOT NULL,
    EnergySectorPerformance FLOAT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- CRE Loan Data
CREATE TABLE CRELoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    PropertyPrices FLOAT NOT NULL,
    VacancyRates FLOAT NOT NULL,
    RentGrowthRates FLOAT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- Mortgage Loan Data
CREATE TABLE MortgageLoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    HomePriceIndex FLOAT NOT NULL,
    ForeclosureRates FLOAT NOT NULL,
    HousingInventory INT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- Construction Loan Data
CREATE TABLE ConstructionLoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    ConstructionSpending FLOAT NOT NULL,
    BuildingPermitsIssued INT NOT NULL,
    CommodityPrices FLOAT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- Retail Loan Data
CREATE TABLE RetailLoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    RetailSalesIndex FLOAT NOT NULL,
    ConsumerSpendingIndex FLOAT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- Consumer Loan Data
CREATE TABLE ConsumerLoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    CreditScoreAverage FLOAT NOT NULL,
    DebtToIncomeRatio FLOAT NOT NULL,
    LoanToValueRatio FLOAT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- M&A Loan Data
CREATE TABLE MALoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    CapitalMarketActivity FLOAT NOT NULL,
    EconomicGrowth FLOAT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);

-- Restructuring Loan Data
CREATE TABLE RestructuringLoanData (
    ID INT PRIMARY KEY IDENTITY(1,1),
    LoanPerformanceID INT NOT NULL,
    Date DATE NOT NULL,
    DefaultRiskRating FLOAT NOT NULL,
    LoanModificationCount INT NOT NULL,
    FOREIGN KEY (LoanPerformanceID) REFERENCES LoanPerformanceData(ID)
);
