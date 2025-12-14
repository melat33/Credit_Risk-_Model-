Credit Risk Probability Model for Alternative Data
An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model
 🏦 Project Overview: Bati Bank Buy-Now-Pay-Later (BNPL) Service
Bati Bank is partnering with a leading eCommerce platform to launch a BNPL service. This project builds a credit risk model using alternative transaction data to score customers with limited formal credit history.

🎯 Core Project Objectives:
1.  Define a proxy variable to label users as high risk or low risk.
2.  Select predictive features correlated with the proxy default variable.
3.  Develop a model that assigns a risk probability (PD) for new customers.
4.  Convert probability into an interpretable credit score.
5.  Recommend optimal loan amount and duration based on customer risk.


 📊 Dataset Summary & Key EDA Insights
Transactions: 95,662 rows
Unique Customers: 3,742
Time Range: 90 days (Nov 2018 - Feb 2019)
Data Quality: 0% missing values, 100% completeness.

🔍 Critical Findings:
1.  Low Fraud Rate: Only 0.20% (193 transactions) are fraudulent. A `FraudResult` proxy for default is insufficient.
2.  Customer RFM Segments: Clustering revealed distinct groups (e.g., 1,119 "At-Risk" customers with low engagement).
3.  High Outlier Correlation: 10% of transactions are outliers, capturing **98.4% of all fraud**—a strong risk signal.
4.  Multicollinearity: High Variance Inflation Factor (VIF) detected in temporal features (e.g., `TransactionStartTime_year` VIF > 170), requiring feature selection.

 🏛️ Credit Scoring Business Understanding

 1. Basel II Accord & Model Interpretability
The Basel II framework** requires banks to hold capital based on quantified risk, making the Probability of Default (PD) a critical, regulated output. This demands:
Interpretable Models: Regulators must understand how decisions are made.
Full Documentation: Complete audit trails for model development and validation.
Stable & Validated Metrics: Models must perform consistently over time.

Our Approach: We prioritize Logistic Regression with Weight of Evidence (WoE) for its high interpretability and regulatory acceptance, while benchmarking against complex models like XGBoost for performance.

 2. The Necessity & Risk of a Proxy Variable
Why a proxy? The dataset has no direct "loan default" label. We must infer risk from behavior.
Our Proxy: RFM (Recency, Frequency, Monetary) analysis clusters customers. The least engaged cluster is labeled `is_high_risk = 1`.

Business Risks:
Proxy Misalignment: Behavioral disengagement (e.g., not buying airtime) may not equal inability to repay a loan.
Unfair Bias: If certain user groups naturally use the platform less, they may be incorrectly labeled high-risk.
Regulatory Scrutiny: Models based on proxies require rigorous validation before live use.

### 3. Model Selection: Interpretability vs. Performance Trade-off
| Model | Accuracy | Interpretability | Regulatory Fit | Best For |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression + WoE | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Core scorecard, regulatory approval |
| XGBoost / LightGBM | High | ⭐ | ⭐ | Benchmarking, capturing non-linear patterns |

Our Strategy: Implement both. Use the interpretable model as the primary scorecard for compliance, and the complex model to validate feature importance and maximum performance potential.

 🔧 Feature Engineering Implementation (Task 3)
All six requirements have been implemented in a reproducible `sklearn` Pipeline.**

Aggregate Features (Requirement 1)
Created for 3,742 unique customers:
   `total_transaction_amount`
   `avg_transaction_amount`
   `std_transaction_amount`
   `transaction_count`
Temporal Features (Requirement 2)
Extracted from `TransactionStartTime`:
Hour, Day, Month, Year, Day of Week
Derived: `is_weekend`, `is_business_hours`

Categorical Encoding (Requirement 3)
One-Hot Encoding** applied to features like `ProductCategory` and `ChannelId`.

 Missing Value Handling (Requirement 4)
   Dataset has 0% missing values. Imputation strategies (e.g., median) are defined in the pipeline for future data.

Normalization/Standardization (Requirement 5)
StandardScaler** used to scale numerical features to mean=0, std=1.

 Weight of Evidence & Information Value (Requirement 6)
High IV Scores Calculated: All aggregate features show suspiciously high IV (>13), indicating very strong separation for the proxy target and requiring review.
Feature Ranking by IV:
    1.  `avg_transaction_amount` (IV = 18.44)
    2.  `total_transaction_amount` (IV = 18.35)
    3.  `transaction_count` (IV = 16.88)
    4.  `std_transaction_amount` (IV = 13.68)

 🚀 Getting Started

 Installation
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd credit-risk-model

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

Next Steps & Roadmap
Model Training: Train and compare Logistic Regression vs. XGBoost models.

Scorecard Development: Calibrate model probabilities to a readable credit score (e.g., 300-850).

Loan Recommendation Engine: Build a rules-based system to suggest loan amounts and terms.

API Deployment: Containerize the model with FastAPI using the provided Dockerfile.

Model Monitoring: Implement drift detection and performance logging with MLflow.