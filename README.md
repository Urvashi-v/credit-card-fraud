# Credit Card Fraud Detection System

## Project Overview
This project presents a **Real-Time Credit Card Fraud Detection System** using **XGBoost** for classification and **Streamlit** for deployment. It handles class imbalance with **SMOTE**, applies **SHAP** for explainability, and uses **interactive gauges** to represent risk levels visually.

The model is trained on the **Credit Card Fraud Detection dataset** available on Kaggle.

---

## Features
- **Automatic dataset download** from KaggleHub
- **Data Preprocessing** (Standard Scaling of Amount)
- **Class Imbalance Handling** using SMOTE
- **XGBoost Classifier** for prediction
- **SHAP** for interpretability and model explanation
- **Real-Time Prediction** with uploaded CSV files
- **Risk Level Gauge Visualization** for high-risk transactions
- **Streamlit Web App** for user interaction

---

## Installation

1. **Clone the repository**
```bash
https://github.com/MohamedAdhil10/credit-card-fraud-detection.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Required Packages:**
- kagglehub
- pandas
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- matplotlib
- joblib
- streamlit
- plotly
- numpy
- streamlit_shap

3. **Kaggle API Setup**
Make sure your Kaggle API token is properly set up to use `kagglehub`.

4. **Run the Application**
```bash
streamlit run app.py
```

---

## How It Works

1. **Training**
   - Downloads the Credit Card dataset.
   - Preprocesses data (standardizes the "Amount" column).
   - Handles imbalance using SMOTE.
   - Trains an XGBoost model.
   - Evaluates using classification report and ROC AUC.
   - Saves the trained model and scaler.

2. **Deployment with Streamlit**
   - Upload transaction CSV.
   - Predicts if each transaction is fraudulent.
   - Displays fraud probability and a simple "Yes/No" fraud detection.
   - Highlights risky transactions and visualizes the top risky transaction with a gauge and SHAP explanations.

---

## Project Structure

```
.
├── README.md             # Project documentation
├── app.py                #  Streamlit application
├── fraud_detection.py    # Training and saving model
├── requirements.txt      # List of required libraries
├── scaler.pkl            # Saved scaler
├── xgb_fraud_model.pkl   # Saved model
└── (creditcard.csv auto-downloaded by kagglehub)                   
```

---

## Example Screenshots

**1. Prediction Results Table**

![Prediction Results Table](https://github.com/user-attachments/assets/4e033672-b23c-4c8e-89c1-4fc99ec75ad9)


**2. SHAP Force Plot**

![SHAP Force Plot](https://github.com/user-attachments/assets/12755d7a-7f57-49ac-a849-313d710fcb9e)


---

## Acknowledgments
- [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## Contact
**Author:** Mohamed Adhil M  
**Email:** adhilm9991@gmail.com

---

Feel free to raise issues, or suggest features!

