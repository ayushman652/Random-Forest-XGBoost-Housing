from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.trainer import train_random_forest, train_xgboost
from src.evaluator import evaluate_model
from src.visualizer import plot_predictions


def main() -> None:

    dataframe = load_data()

    print(f"Dataset shape: {dataframe.shape}")
    print(f"Missing values: {dataframe.isna().sum().sum()}")

    # Prepare training and testing data
    X_train, X_test, y_train, y_test = preprocess_data(dataframe)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train Random Forest
    rf_model, rf_training_time = train_random_forest(
        X_train,
        y_train,
    )

    # Train XGBoost
    xgb_model, xgb_training_time = train_xgboost(
        X_train,
        y_train,
    )

    # Evaluate Random Forest
    rf_predictions, rf_prediction_time, rf_mse, rf_r2 = evaluate_model( rf_model, X_test, y_test,)

    # Evaluate XGBoost
    xgb_predictions, xgb_prediction_time, xgb_mse, xgb_r2 = evaluate_model( xgb_model, X_test, y_test, )

    # Display results
    print("\nRandom Forest")
    print(f"Training time: {rf_training_time:.4f} seconds")
    print(f"Prediction time: {rf_prediction_time:.4f} seconds")
    print(f"MSE: {rf_mse:.4f}")
    print(f"R²: {rf_r2:.4f}")

    print("\nXGBoost")
    print(f"Training time: {xgb_training_time:.4f} seconds")
    print(f"Prediction time: {xgb_prediction_time:.4f} seconds")
    print(f"MSE: {xgb_mse:.4f}")
    print(f"R²: {xgb_r2:.4f}")

    # Create visualizations
    plot_predictions(
        y_test,
        rf_predictions,
        "Random Forest",
    )

    plot_predictions(
        y_test,
        xgb_predictions,
        "XGBoost",
    )


if __name__ == "__main__":
    main()