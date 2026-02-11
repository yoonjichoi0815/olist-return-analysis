# Customer Return Analysis using Olist E-Commerce Dataset

## Project Overview
This project analyzes product return behavior in Brazilian e-commerce data (from Olist) to identify key factors associated with product returns. It explores the relationship between customer review scores and return status, applies machine learning models to predict returns, and leverages SHAP for interpretability. Additionally, NLP techniques are used to analyze customer review texts.

## Project Goals
1. Analyze the relationship between customer review scores and return rates.
2. Identify product categories with higher return tendencies.
3. Develop machine learning models to predict product returns.
4. Interpret model outputs using SHAP to identify key features.
5. Explore customer sentiments through NLP on review text.

## Key Questions
1. Are lower review scores associated with higher return rates?
2. Do certain product categories have higher return rates?
3. Can return status be predicted based on review scores, delivery information, and payment details?

## Methodology
- **Data Preprocessing**: Merging datasets to create a unified analysis table
- **Exploratory Data Analysis (EDA)**: Descriptive statistics, visualizations, correlation analysis
- **Statistical Testing**:
  - T-test between low and high score groups
  - ANOVA across product categories
  - Chi-square test of independence
- **Baseline Modeling**:
  - Simple & Multiple Linear Regression (e.g., using `review_score`, `delivery_late`, `total_price`)
- **Modeling**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SMOTE for class imbalance handling  

## Key Findings
- A weak negative correlation was found between review score and return status (r = -0.12)
- T-tests and regression confirmed a significant difference in return rates between low and high review scores
- Chi-square tests revealed that product category is significantly related to return rates

## Model Performance Summary (After SMOTE)
- Logistic model:
  - **Accuracy**: 88%
  - **Recall (for returns)**: 79%
  - **Precision**: 3%
  - **F1-score**: 6%
- Random Forest model:
  - **Accuracy**: 95%
  - **Recall (for returns)**: 57%
  - **Precision**: 6%
  - **F1-score**: 10%
- XGBoost model:
  - **Accuracy**: 88%
  - **Recall (for returns)**: 76%
  - **Precision**: 3%
  - **F1-score**: 6%

- Among all models, **Random Forest** showed the best trade-off between recall and precision, achieving the highest **F1-score** for identifying returned products.
- While Logistic Regression and XGBoost achieved higher recall, they had significantly lower precision.

## Tools & Libraries
- Python (Pandas, NumPy, Seaborn, Scikit-learn, Statsmodels, imbalanced-learn, XGBoost, SHAP)
- Streamlit for model explanation interface
- NLTK, TextBlob for NLP analysis

## 📂 Project Files
- `olist_return_analysis.ipynb`: Main notebook for structured data analysis, hypothesis testing, and modeling.
- `review_nlp_analysis.ipynb`: NLP analysis of customer review texts (translation, sentiment analysis, etc.).
- `app.py`: Streamlit-based interactive SHAP visualization for model interpretation.

## Dataset Source
[Kaggle - Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Author
Yoonji Choi  
Contact: [yoonjichoi0815@gmail.com]  
GitHub: [https://github.com/yoonjichoi0815]

**This project is intended for portfolio demonstration purposes only. Please do not copy or redistribute without permission.**