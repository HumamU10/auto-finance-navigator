
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data setup (used for simple forecasting)
# In production, you'd load a cleaned dataset or trained model
df = pd.read_csv("student_spending (1).csv")
df = df.dropna(subset=['monthly_income', 'transportation'])

expense_columns = ['housing', 'food', 'transportation', 'books_supplies',
                   'entertainment', 'personal_care', 'technology',
                   'health_wellness', 'miscellaneous']
df['total_monthly_expenses'] = df[expense_columns].sum(axis=1)
df['savings_potential'] = df['monthly_income'] - df['total_monthly_expenses']

# Train a simple model for demonstration
X = df[['monthly_income', 'transportation']]
y = df['savings_potential']
model = LinearRegression()
model.fit(X, y)

# --- Streamlit App ---
st.set_page_config(page_title="AutoFinance Navigator", layout="centered")

st.title("🚗 AutoFinance Navigator")
st.subheader("Student Car Budget & Savings Forecast")

# User Inputs
with st.form(key='input_form'):
    monthly_income = st.number_input("Monthly Income (€)", min_value=0, value=1000)
    transport_cost = st.number_input("Monthly Transport Cost (€)", min_value=0, value=200)
    submitted = st.form_submit_button("Generate Forecast")

if submitted:
    # Predict current savings
    predicted_savings = model.predict([[monthly_income, transport_cost]])[0]

    st.success(f"Estimated Monthly Savings: €{predicted_savings:.2f}")

    # Forecast savings over 12 months
    months = pd.date_range(start='2025-01-01', periods=12, freq='M')
    variation = np.random.normal(loc=0, scale=20, size=12)
    forecast_savings = predicted_savings - variation

    # Plot forecast
    fig, ax = plt.subplots()
    ax.plot(months, forecast_savings, marker='o', linestyle='-', color='green')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("📈 Forecasted Savings Over 12 Months")
    ax.set_xlabel("Month")
    ax.set_ylabel("Forecasted Savings (€)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Optional tip
    if predicted_savings < 0:
        st.warning("⚠️ Your expenses exceed your income. Consider reducing transport costs or exploring financial aid.")
    else:
        st.info("✅ You're in a healthy savings position. Keep it up!")
