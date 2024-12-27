import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading the trained model

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Streamlit app title
st.title("Gold Price Prediction App")

# Input fields for feature values
st.sidebar.header("Input Features")
spx = st.sidebar.number_input("S&P 500 Index (SPX)", value=1650.0, step=1.0)
uso = st.sidebar.number_input("Crude Oil Price (USO)", value=30.0, step=1.0)
slv = st.sidebar.number_input("Silver Price (SLV)", value=20.0, step=0.1)
eur_usd = st.sidebar.number_input("EUR/USD Exchange Rate", value=1.3, step=0.01)

# Predict button
if st.button("Predict Gold Price"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'SPX': [spx],
        'USO': [uso],
        'SLV': [slv],
        'EUR/USD': [eur_usd]
    })

    # Make a prediction
    prediction = model.predict(input_data)
    st.success(f"Predicted Gold Price: ${prediction[0]:.2f}")
