# Credit Card Default Risk Prediction

**Capstone Project | Python, Scikit-learn, XGBoost, SMOTE, SHAP**  
**Feb 2024**

## Project Overview
This project focuses on predicting customer default risk using the **Taiwan credit card dataset** (30,000 records, 25 features). The goal is to build an accurate classification model to identify customers likely to default on their credit payments.  

## Key Contributions
- **Data Preprocessing**:  
  - Cleaned anomalies in `education` and `marriage` categories  
  - Handled multicollinearity among features  
  - Encoded categorical variables for modeling  

- **Class Imbalance Handling**:  
  - Original dataset had 22% defaulters  
  - Applied **SMOTETomek oversampling** to improve model recall  

- **Modeling**:  
  - Trained and compared multiple models:  
    - Logistic Regression  
    - XGBoost  
    - Random Forest  
    - K-Nearest Neighbors (KNN)  
    - Extra Trees  

- **Model Interpretation**:  
  - Used **SHAP values** to interpret feature importance and understand model decisions  

## Key Results
- Improved detection of defaulters using oversampling techniques  
- XGBoost and Random Forest showed strong predictive performance  
- Insights from SHAP provided actionable understanding of important factors affecting default risk  

## Tools & Libraries
- **Python**  
- **Scikit-learn**  
- **XGBoost**  
- **SMOTE (imbalanced-learn)**  
- **SHAP**  
- **Pandas, NumPy, Matplotlib, Seaborn**  

## Usage
1. Clone the repository:  
```bash
git clone https://github.com/tanisha2203/<repo-name>.git



