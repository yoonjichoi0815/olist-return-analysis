# Customer Return Prediction & Review Analysis (Olist Dataset)

## Project Overview
This project analyzes product return behavior in Brazilian e-commerce data (from Olist) to identify key factors associated with product returns. It explores the relationship between customer review scores and return status, applies machine learning models to predict returns, and leverages SHAP for interpretability. Additionally, NLP techniques are used to analyze customer review texts.

## Project Overview
This project investigates product return behavior in e-commerce using the Brazilian Olist dataset.

It combines:
- Structured data modeling (price, delivery, review score)
- Text analysis (NLP) on customer reviews
- Explainable AI (SHAP) for interpretation

Goal: Understand why customers return products and build a predictive model with interpretable insights.

## Key Contributions
- Built an end-to-end data pipeline (EDA → statistical testing → ML → NLP → deployment)
- Handled class imbalance (rare returns) using SMOTE
- Integrated text features (TF-IDF) with structured data
- Applied interpretable ML (SHAP + coefficient analysis)
- Deployed results via Streamlit dashboard:
  - Global feature importance
  - Local prediction explanation

## Methodology
1. **Data Processing**
- Merged multiple relational tables into a unified dataset
- Engineered features:
  - delivery_late
  - total_price
  - review_score

2. **Exploratory Data Analysis (EDA)**
- Descriptive statistics
- visualizations
- correlation analysis

3. **Statistical Testing**
  - T-test between low and high score groups
  - ANOVA across product categories
  - Chi-square test of independence

4. **Predictive Modeling**
- Baseline Modeling:
  - Simple & Multiple Linear Regression (e.g., using `review_score`, `delivery_late`, `total_price`)
- Modeling:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SMOTE for class imbalance handling  

5. **NLP Analysis**
- Translated Portuguese reviews → English
- TF-IDF vectorization
- Combined: structured features + text features
- Extracted:
  - Top words in returned vs non-returned items
  - Negative word impact on return rate
  - Model-based word importance

## Key Findings
- A weak negative correlation was found between review score and return status (r = -0.12)
- T-tests and regression confirmed a significant difference in return rates between low and high review scores
- Chi-square tests revealed that product category is significantly related to return rates
- Words like "delay", "bad", "terrible" strongly increase return probability
- Positive sentiment words ("good", "excellent") strongly correlate with non-returns
- Text features significantly improve predictive performance

## Model Performance Summary (After SMOTE)
- Logistic model:
  - Accuracy: 88%
  - Recall (for returns): 79%
  - Precision: 3%
  - F1-score: 6%
- Random Forest model:
  - Accuracy: 95%
  - Recall (for returns): 57%
  - Precision: 6%
  - F1-score: 10%
- XGBoost model:
  - Accuracy: 88%
  - Recall (for returns): 76%
  - Precision: 3%
  - F1-score: 6%

## NLP Model (Structured + Text) Performance
- Logistic Regression (interpretable model):
  - Recall (returns): 0.91
  - F1-score: 0.71
  - PR-AUC: 0.70

- Among all models, **Random Forest** showed the best trade-off between recall and precision, achieving the highest **F1-score** for identifying returned products.
- While Logistic Regression and XGBoost achieved higher recall, they had significantly lower precision.
- NLP model shows significant improvement after adding text features.

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- NLP: TF-IDF, text preprocessing
- Imbalanced Learning: SMOTE
- Explainability: SHAP
- Visualization: Seaborn, Matplotlib
- Deployment: Streamlit

## Dataset Source
[Kaggle - Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Author
Yoonji Choi  
Contact: [yoonjichoi0815@gmail.com]  
GitHub: [https://github.com/yoonjichoi0815]

**This project is intended for portfolio demonstration purposes only. Please do not copy or redistribute without permission.**