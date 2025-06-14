import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import base64
import pickle
import joblib

# --- Title and Intro ---
st.title("💳 UPI Transaction Fraud Detector")

st.markdown("""
You can inspect a single transaction manually **OR** upload a `.csv` file to check multiple transactions at once.
""")

# --- Load Model ---
pickle_file_path = "UPI Fraud Detection updated.pkl"
loaded_model = pickle.load(open(pickle_file_path, 'rb'))
expected_features = joblib.load(open('model_features.pkl', 'rb'))

# --- Categorical Options ---
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
      'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
      'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 'More Services',
      'Other', 'Purchases', 'Travel bookings', 'Utilities']

# --- Manual Input ---
st.subheader("📌 Single Transaction Input")
tran_date = st.date_input("Transaction Date", datetime.date.today())
selected_date = dt.combine(tran_date, dt.min.time())
month = selected_date.month
year = selected_date.year

tran_type = st.selectbox("Transaction Type", tt)
pmt_gateway = st.selectbox("Payment Gateway", pg)
tran_state = st.selectbox("Transaction State", ts)
merch_cat = st.selectbox("Merchant Category", mc)
amt = st.number_input("Transaction Amount", step=0.1)

st.markdown("---")
st.subheader("📁 Upload CSV for Bulk Detection")
uploaded_file = st.file_uploader("Upload your `.csv` file", type=["csv"])

# Show sample format
if st.checkbox("Show expected CSV format"):
    sample = pd.DataFrame({
        "Amount": [499.0],
        "Date": ["12-2023"],
        "Transaction_Type": ["Bill Payment"],
        "Payment_Gateway": ["Google Pay"],
        "Transaction_State": ["Karnataka"],
        "Merchant_Category": ["Purchases"]
    })
    st.write(sample)

# --- Button Logic ---
if st.button("Check Transaction(s)"):
    if uploaded_file is not None:
        with st.spinner("Checking uploaded transactions..."):
            df = pd.read_csv(uploaded_file)

            # Split 'Date' into Month and Year
            df[['Month', 'Year']] = df['Date'].str.split('-', expand=True)[[0, 1]].astype(int)
            df.drop(columns=['Date'], inplace=True)

            # One-hot encode
            df_encoded = pd.get_dummies(df)

            # Align with training columns
            df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)

            # Predict
            df['Fraud_Predicted'] = loaded_model.predict(df_encoded)

            st.success("✅ Completed fraud detection for all transactions.")
            st.dataframe(df)

            # Download link
            def download_csv(dataframe):
                csv = dataframe.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="fraud_detection_output.csv">📥 Download Result CSV</a>'

            st.markdown(download_csv(df), unsafe_allow_html=True)

    else:
        with st.spinner("Checking single transaction..."):
            input_df = pd.DataFrame([{
                'Amount': amt,
                'Year': year,
                'Month': month,
                'Transaction_Type': tran_type,
                'Payment_Gateway': pmt_gateway,
                'Transaction_State': tran_state,
                'Merchant_Category': merch_cat
            }])

            # One-hot encode
            input_df = pd.get_dummies(input_df)

            # Align with training features
            input_df = input_df.reindex(columns=expected_features, fill_value=0)

            # Predict
            result = loaded_model.predict(input_df)[0]

            st.success("✅ Transaction Checked!")
            if result == 0:
                st.write("🎉 This transaction is **NOT fraudulent**.")
            else:
                st.error("⚠️ Warning! This transaction is **fraudulent**.")
