import time

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(
    model,
    X_test,
    y_test,
) -> tuple[np.ndarray, float, float, float]:
    """
    Generate predictions and evaluate a regression model.

    Returns:
        Tuple containing:
        - predictions
        - prediction time
        - mean squared error
        - R² score
    """
    start_time = time.time()
    predictions = model.predict(X_test)
    prediction_time = time.time() - start_time

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return predictions, prediction_time, mse, r2