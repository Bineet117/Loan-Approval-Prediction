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
    data =  joblib.load(Config.DATA_PATH)
    return data

def load_model():
    return joblib.load(Config.MODEL_PATH)

# User input function
def user_input_features(df):
    # Input features from the user
    dependency = st.selectbox("No_of_dependents", sorted(df["no_of_dependents"].value_counts().index.values))
    education = st.selectbox("Education", df["education"].unique())
    self_employed = st.selectbox("Self Employed", df["self_employed"].unique())
    applicant_income = st.number_input("Applicant Income", min_value=0, max_value=9900000, value=50000)  # Average applicant income
    loan_amount = st.number_input("Loan Amount", min_value=0, max_value=39500000, value=200000)  # Average loan amount
    loan_term = st.selectbox("Loan Term", sorted(df["loan_term"].value_counts().index.values))
    cibil_score = st.number_input("Cibil Score", min_value=300, max_value=900, value=650)  # Average CIBIL score
    total_asset = st.number_input("Total Asset ('residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value')", 
                                  min_value=600000, max_value=107400000, value=5000000)  # Average total assets
    
    # Updated user_data dictionary to match the user inputs
    user_data = {
        'No_of_dependents': dependency,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'LoanAmount': loan_amount,
        'LoanTerm': loan_term,
        'CibilScore': cibil_score,
        'TotalAsset': total_asset
    }
    
    # Convert the user data into a DataFrame
    # features = pd.DataFrame(user_data, index=[0])
    inputs = pd.DataFrame([[dependency,education,self_employed,applicant_income,loan_amount,loan_term,cibil_score,total_asset]], columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score','total_asset_value'])
    return inputs



# Main function
def main():
    st.title("Loan Approval Prediction App") 
    
    data = load_data()
    input_df = user_input_features(data)

    st.subheader('User Input Data')
    st.write(input_df)

    # Load the trained model
    pipeline = load_model()

    # Make predictions
    prediction = pipeline.predict(input_df)[0]

    st.subheader('Prediction')
    def final_pred(prediction):
        if prediction == 1:
            return "Approved"
        else:
            return "Rejected"
    result = final_pred(prediction=prediction)
    st.write(f"Loan Status: **{result}**")

if __name__ == "__main__":
    main()