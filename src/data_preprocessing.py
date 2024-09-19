import pandas as pd
import numpy as np
import joblib

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess(self,df):
        df.drop("loan_id" , axis = 1, inplace = True)
        # removing extra space from each column name
        df.columns = [cols.strip() for cols in df.columns]
        df = df[df["residential_assets_value"] > 0]
        # creating new colummn `total_asset_value`
        df["total_asset_value"] = df['residential_assets_value'] + df['commercial_assets_value'] + df['luxury_assets_value'] + df['bank_asset_value']
        # dropping these columns - "residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"
        df.drop(columns = ["residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"], axis = 1)
        return df

    def mapping(self, data):
        # removing space from the data and mapping
        data["education"] = data["education"].apply(lambda x:x.strip())
        # data["education"] = data["education"].map({"Graduate":1,"Not Graduate":0 })

        # removing space from the data and mapping
        data["self_employed"] = data["self_employed"].apply(lambda x:x.strip())
        # data["self_employed"] = data["self_employed"].map({"Yes":1,"No":0 })

        # removing space from the data and mapping
        data["loan_status"] = data["loan_status"].apply(lambda x:x.strip())
        data["loan_status"] = data["loan_status"].map({"Approved":1,"Rejected":0 })
        return data

    def save_processed_data(self,data, DATA_PATH):
            joblib.dump(data,DATA_PATH)


