import sys
import os
from sklearn.metrics import r2_score

# Add the 'src' folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_ingestion import Dataloader
from src.data_preprocessing import DataPreprocessor
from src.model_building import LoanApprovalModel
from src.config import Config

def main():
    # Load data
    data_loader = Dataloader()
    df = data_loader.load_raw_data()

    # Preprocess data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    df_clean = preprocessor.mapping(df_processed)

    # save cleaned data
    preprocessor.save_processed_data(df_clean,Config.DATA_PATH)

    # Train model
    model = LoanApprovalModel()  # No need to pass preprocessor
    column_trans = model.columntransformer()
    pipeline = model.create_pipeline(column_trans)
    X = df_clean.drop(columns=[Config.TARGET_COLUMN], axis=1)
    y = df_clean[Config.TARGET_COLUMN]
    split = model.split_data(X=X, y=y)

    # Fit model
    X_train, X_test, y_train, y_test = split
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)
    print(f"predicted score: {score:.4f}")  # Format the score with 4 decimal places

    # Save model
    model.save_model(pipeline, Config.MODEL_PATH)

if __name__ == "__main__":
    main()
