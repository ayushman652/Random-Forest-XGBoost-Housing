import pandas as pd

from .config import DATASET_PATH


def load_data() -> pd.DataFrame:
   
    return pd.read_csv(DATASET_PATH)