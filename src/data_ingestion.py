import pandas as pd
from config import Config

class Dataloader:
    def __init__(self):
        self.raw_data_path = Config.RAW_DATA_PATH


    def load_raw_data(self):
        return pd.read_csv(self.raw_data_path)
        
