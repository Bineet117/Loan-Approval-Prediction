from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

class LoanApprovalModel:
    def __init__(self):
        pass
    
    def columntransformer(self):
        column_transformed = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['income_annum', 'loan_amount', 'cibil_score','total_asset_value']),
            ('cat', OneHotEncoder(), ['no_of_dependents', 'education', 'self_employed', 'loan_term'])
        ])
        return column_transformed


    def create_pipeline(self,column_transformed):
        pipeline = Pipeline([
        ('preprocessor', column_transformed),  # Preprocessing step
        ('classifier', DecisionTreeClassifier())  # Machine learning model
        ])
        return pipeline

    def save_model(self, pipeline, model_path):
        joblib.dump(pipeline, model_path)

    def load_model(self, model_path):
        return joblib.load(model_path)

    def split_data(self, X, y, test_size=0.25):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test
