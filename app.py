import streamlit as st
import joblib
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Add the 'src' folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import Config


def load_data():
    return joblib.load(Config.DATA_PATH)

def load_model():
    return joblib.load(Config.MODEL_PATH)

# Sidebar for user input
def user_input_features(df):
    st.sidebar.header("Enter Loan Details")
    
    dependency = st.sidebar.selectbox("Number of Dependents", sorted(df["no_of_dependents"].value_counts().index.values))
    education = st.sidebar.radio("Education", df["education"].unique())
    self_employed = st.sidebar.radio("Self Employed", df["self_employed"].unique())

    applicant_income = st.sidebar.number_input("Applicant Income (₹)", min_value=0, max_value=9900000, value=50000, step=5000)
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, max_value=39500000, value=200000, step=50000)
    loan_term = st.sidebar.selectbox("Loan Term (Months)", sorted(df["loan_term"].value_counts().index.values))

    cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900, value=650, step=10)
    
    total_asset = st.sidebar.number_input(
        "Total Asset Value (₹)", min_value=30000, max_value=107400000, value=50000, step=500000
    )

    if applicant_income < 20000:
        st.sidebar.warning("⚠️ Low income may reduce approval chances.")
    if loan_amount > 5000000:
        st.sidebar.warning("⚠️ High loan amounts require a strong credit profile.")
    if cibil_score < 600:
        st.sidebar.warning("⚠️ A low CIBIL score may lead to rejection.")

    inputs = pd.DataFrame(
        [[dependency, education, self_employed, applicant_income, loan_amount, loan_term, cibil_score, total_asset]], 
        columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                 'loan_amount', 'loan_term', 'cibil_score', 'total_asset_value']
    )
    
    return inputs


# Main function
def main():
    st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")

    st.title("🏦 Loan Approval Prediction")
    st.image("static/images/Instant-Loan-Approval[1].png", use_column_width=True)

    data = load_data()
    input_df = user_input_features(data)

    st.subheader("🔍 User Input Summary")
    st.write(input_df)

    if st.button("Predict Loan Status"):
        pipeline = load_model()
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][prediction]

        # Loan Approval Result
        st.subheader("📌 Loan Prediction Result")
        
        if prediction == 1:
            st.success(f"✅ Approved")
        else:
            st.error(f"❌ Rejected")
        
        # Tips Section
        if prediction == 0:
            st.info("💡 **Tips to Improve Approval Chances:**\n"
                    "- Increase your CIBIL score (Above 700 is preferred)\n"
                    "- Opt for a lower loan amount or longer tenure\n"
                    "- Maintain a strong asset portfolio")

if __name__ == "__main__":
    main()
