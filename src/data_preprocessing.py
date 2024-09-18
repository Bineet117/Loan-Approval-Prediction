import pandas as pd
import numpy as np

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
        return df

    def mapping(self, data):
        # removing space from the data and mapping
        data["education"] = data["education"].apply(lambda x:x.strip())
        data["education"] = data["education"].map({"Graduate":1,"Not Graduate":0 })

        # removing space from the data and mapping
        data["self_employed"] = data["self_employed"].apply(lambda x:x.strip())
        data["self_employed"] = data["self_employed"].map({"Yes":1,"No":0 })

        # removing space from the data and mapping
        data["loan_status"] = data["loan_status"].apply(lambda x:x.strip())
        data["loan_status"] = data["loan_status"].map({"Approved":1,"Rejected":0 })
        return data


# # Assume we have a dataframe `df`
# df = pd.DataFrame({
#     'loan_id': [1, 2, 3, 4],
#     'residential_assets_value': [5000, 0, 3000, 4500],
#     'commercial_assets_value': [2000, 1500, 3000, 1000],
#     'luxury_assets_value': [1000, 500, 200, 300],
#     'bank_asset_value': [1500, 2000, 1500, 1300],
#     'education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
#     'self_employed': ['Yes', 'No', 'Yes', 'No'],
#     'loan_status': ['Approved', 'Rejected', 'Approved', 'Rejected']
# })

# preprocessor = DataPreprocessor()
# df_processed = preprocessor.preprocess(df)
# df_mapped = preprocessor.mapping(df_processed)

# print(df_mapped)
