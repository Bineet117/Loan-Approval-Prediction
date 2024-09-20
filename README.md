
# 🏦🏦 Loan Approval Prediction 🏦🏦

## 💰 Introduction
This project focuses on building a predictive model to determine whether a loan application will be approved or rejected. By analyzing various features such as applicant income, credit score, loan amount, and other relevant factors, the model uses machine learning techniques to classify loan applications based on past data. The goal is to help financial institutions make faster, data-driven decisions while minimizing the risk of default. This project includes data preprocessing, feature engineering, model selection, and evaluation to ensure accurate predictions.

## 💰 Dataset Description
## Dataset Description:

💸 **loan_id**: A unique identifier for each loan application.

💸 **no_of_dependents**: The number of dependents (children, spouse, parents, etc.) that the loan applicant financially supports. This can affect their ability to repay the loan.
   - Example: `1` means the applicant has 1 dependent; `2` means the applicant has 2 dependents.

💸 **education**: The education level of the loan applicant (e.g., Graduate, Not Graduate). Education can be a factor in determining the applicant's employability and income potential.
   - Example: `Graduate` or `Not Graduate`.

💸 **self_employed**: A binary indicator (Yes/No) that shows whether the applicant is self-employed. Self-employed individuals may have irregular income, which could affect loan approval chances.
   - Example: `Yes` if self-employed, `No` otherwise.

💸 **income_annum (in lakhs)**: The annual income of the loan applicant in lakhs. Higher income increases the applicant's ability to repay the loan.
   - Example: `96,00,000` translates to 96 lakhs.

💸 **loan_amount (in lakhs)**: The amount of money the applicant is requesting as a loan, in lakhs. Higher loan amounts can lead to stricter approval criteria.
   - Example: `2,99,00,000` translates to 299 lakhs.

💸 **loan_term (in years)**: The duration of the loan, typically in years. A longer loan term may reduce the monthly repayment amount but increase the overall interest paid.
   - Example: `12` means a 12-year loan term.

💸 **cibil_score**: The credit score of the loan applicant, typically a score ranging from 300 to 900 that reflects the applicant’s creditworthiness. Higher scores indicate lower risk for the lender.
   - Example: `778` is a high score, indicating good credit.

💸 **residential_assets_value (in lakhs)**: The monetary value of any residential property or assets the applicant owns, in lakhs. These assets could be used as collateral or increase the applicant’s perceived ability to repay.
   - Example: `24,00,000` translates to 24 lakhs.

💸 **commercial_assets_value (in lakhs)**: The monetary value of any commercial property or assets the applicant owns, in lakhs. This can further enhance the applicant’s financial standing and improve loan approval chances.
    - Example: `1,76,00,000` translates to 176 lakhs.

💸 **luxury_assets_value (in lakhs)**: The value of any luxury assets owned by the applicant (e.g., cars, jewelry, etc.), in lakhs. Luxury assets can contribute to the overall financial profile of the applicant.
    - Example: `2,27,00,000` translates to 227 lakhs.

💸 **bank_asset_value (in lakhs)**: The value of assets or funds the applicant holds in bank accounts, in lakhs. This may serve as evidence of liquidity or savings that could cover loan payments.
    - Example: `80,00,000` translates to 80 lakhs.

💸 **loan_status**: The outcome of the loan application (e.g., Approved or Rejected). This is the target variable for prediction in a loan approval model. The goal is to predict whether a loan will be approved based on the applicant's profile.
    - Example: `Approved` or `Rejected`.

[**🗂️ DATASET LINK**](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)


## 💰 Installation

Clone the repository:

```bash
   git clone https://github.com/Bineet117/Activity1-Loan-Approval-Prediction.git
```

To install required libraries:

```bash
  pip install -r requirements.txt
```
 
## 💰 Usage
To run the Streamlit application:

```bash
  streamlit run app.py
```
Select the features for loan Prediction.
## Technologies_Used

