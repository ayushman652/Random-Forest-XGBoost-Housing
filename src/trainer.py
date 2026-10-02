import time

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .config import RANDOM_STATE, RF_N_ESTIMATORS, XGB_N_ESTIMATORS


def train_random_forest( X_train, y_train,) -> tuple[RandomForestRegressor, float]:
    """
    Train a Random Forest regression model.
    Returns:
        Tuple containing the trained model and training time.
    """
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    return model, training_time


def train_xgboost( X_train, y_train,) -> tuple[XGBRegressor, float]:
    """
    Train an XGBoost regression model.
    Returns:
        Tuple containing the trained model and training time.
    """
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    return model, training_time