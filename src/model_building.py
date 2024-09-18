from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import joblib

class LoanApprovalModel:
    def __init__(self):
        pass

    def create_pipeline(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        return pipeline

    def save_model(self, pipeline, model_path):
        joblib.dump(pipeline, model_path)

    def load_model(self, model_path):
        return joblib.load(model_path)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
