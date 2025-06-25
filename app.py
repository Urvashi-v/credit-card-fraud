import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import streamlit_shap

# Load model and scaler
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Real-Time Credit Card Fraud Detection System")

uploaded_file = st.file_uploader("Upload a transaction file (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess uploaded data
    data = pd.read_csv(uploaded_file)

    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])

    if 'Amount' in data.columns:
        data['Amount'] = scaler.transform(data[['Amount']])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    data['Fraud Probability'] = probability
    data['Is Fraudulent'] = np.where(prediction == 1, 'Yes', 'No')

    st.subheader("Full Prediction Results")
    st.dataframe(data)

    st.subheader("Flagged Risky Transactions")
    risky = data[data['Is Fraudulent'] == 'Yes']
    st.dataframe(risky)

    if not risky.empty:
        st.subheader("SHAP Explanation and Risk Level for Top Risky Transaction")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data.drop(['Fraud Probability', 'Is Fraudulent'], axis=1))

        # Get Top Risky Transaction
        top_idx = risky['Fraud Probability'].idxmax()
        top_transaction = data.drop(['Fraud Probability', 'Is Fraudulent'], axis=1).loc[top_idx]
        top_probability = risky.loc[top_idx, 'Fraud Probability']


        # Risk Level Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=top_probability * 100,
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 8},
                    'thickness': 0.9,
                    'value': top_probability * 100
                }
            },
            title={'text': "Fraud Risk (%)"}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP Force Plot (Corrected)
        st.subheader("SHAP Force Plot Explanation")

        shap.initjs()
        streamlit_shap.st_shap(
            shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                top_transaction
            ),
            height=300
        )

    else:
        st.info("No fraudulent transactions detected!")
else:
    st.info("Please upload a transaction CSV file to proceed.")
