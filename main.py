import sys
import os


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

    # Train model
    model = LoanApprovalModel()  # No need to pass preprocessor
    pipeline = model.create_pipeline()
    X = df_clean.drop(columns=[Config.TARGET_COLUMN], axis=1)
    y = df_clean[Config.TARGET_COLUMN]
    split = model.split_data(X=X, y=y)

    # Fit model
    X_train, X_test, y_train, y_test = split
    pipeline.fit(X_train, y_train)

    # Save model
    model.save_model(pipeline, Config.MODEL_PATH)

if __name__ == "__main__":
    main()
