import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def preprocess_data( dataframe: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test