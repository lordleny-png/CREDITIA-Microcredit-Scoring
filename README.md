# CREDITIA-Microcredit-Scoring
An explainable credit risk assessment framework for microfinance institutions in West Africa
## Overview
CREDITIA is a data science project designed to support microcredit risk assessment by combining expert-based scoring, financial capacity analysis, regulatory eligibility checks, and machine learning.
The project explores how data analytics and predictive modeling can help microfinance institutions assess repayment risk, estimate sustainable loan amounts, and support more consistent lending decisions.
The initial scope focuses on five west african countries: Benin,Burkina-Faso,Mali, Senegal and Togo.
 Project status: Prototype / Research and Development 
## Objectives
- Assess borrowers' repayment capacity
- Combine quantitative and qualitave credit scoring
- Estimate the probability of default using XGBoost
- Identify applications requiring additional review
- Integrate eligibility and regulatory control checks into the assessment workflow
- Provide a foundation for a potential Streamlit decision-support application
## Key Features
## 1. Expert-Based Credit Scoring
The proposed scoring framework combines two components:

Quantitative score — 60 points: repayment capacity, repayment regularity, existing debt, repayment history and credit information, and savings.

Qualitative score — 40 points: business or employment continuity, loan-purpose consistency, financial management discipline, resilience, market stability, contact stability, and available guarantees or support.

The resulting expert score is expressed on a 100-point scale.

## 2. Repayment Capacity Assessment
The framework distinguishes between three income profiles:

Salaried employees and pensioners

Self-employed borrowers and business owners

Seasonal or mixed-income borrowers

Capacity calculations account for income, business costs where applicable, essential household expenses, existing debt obligations, and estimated minimum living expenses.

The resulting estimate is intended to support the assessment of an affordable installment and a sustainable loan amount.

## 3. Machine Learning
The project uses XGBoost to explore the prediction of credit default.

The target variable is default_90d, defined as whether a borrower reaches 90 or more days past due during the specified observation window.

The preprocessing workflow includes numerical imputation and categorical-variable encoding.

The model is intended to complement, not replace, expert assessment and mandatory eligibility controls.

## 4. Risk Classification
The proposed expert rating scale includes:
A+ : 95-100,| A=90-94.99, |  A-= 85-89.99 , | B+= 80-84.99 , | B= 75-79.99, | B-=70-74.99, | C+=65-69.99, | C=60-64.99, | C-=55-59.99, | D+= 50-54.99, | D=45-49.99, | D-=40-44.99, | E= Below 40|.
These ratings are prototype scoring thresholds, not official regulatory credit classifications. They should be validated against actual repayment outcomes before operational use.

## 5. Eligibility and Risk Controls
The assessment workflow includes checks for:

Know Your Customer (KYC) completeness

Credit information consent and consultation

Fraud indicators and major inconsistencies

Loan-purpose eligibility

Legal eligibility and loan-term constraints

Serious active arrears

Repayment capacity

Required membership contributions being up to date, where applicable

Mandatory controls are applied independently of the machine learning prediction. A low predicted default probability must not override a mandatory block.

Regulatory references and thresholds must be verified against applicable official texts before deployment.

## Data
The development dataset supplied for this prototype contains synthetic observations.

Synthetic data is useful for testing data pipelines, feature engineering, scoring logic, and model integration. However, it does not establish real-world credit risk performance.

The dataset may include application characteristics, financial variables, repayment history, expert scores, eligibility indicators, and the default_90d target.

No real borrower-level information should be published without appropriate authorization and privacy safeguards.

Model Evaluation
Model evaluation should include:

ROC-AUC

Precision-Recall AUC

Precision, recall, and F1-score

Confusion matrix

Comparison with a logistic regression baseline

Comparison with the expert-based score

Temporal validation and probability calibration

A preliminary ROC-AUC of 0.6664 was previously reported for a prototype trained on synthetic data. This result is indicative only and must not be interpreted as evidence of real-world predictive performance.

Technology Stack
Python

Pandas and NumPy

Scikit-learn

XGBoost

Streamlit (for the user interface)

Matplotlib / Seaborn (for visualizations, where used)
Project Structure
CREDTIA-Microcredit-Scoring/
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── data/
├── models/
│   ├── xgboost.pkl
│   └── median.pkl
├── notebooks/
└── src/
    ├── scoring.py
    ├── capacity.py
    └── regulation.py
This is the intended structure; only files that exist in the repository should be listed as implemented components.
## Disclaimer
This project is intended for educational, analytical, and prototyping purposes. It does not constitute financial, legal, or regulatory advice. The scoring thresholds and modeled assumptions are not official regulatory standards.

## Author
Leny Napala

Data Analytics | Statistical Modeling | Machine Learning | Credit Risk Analytics














