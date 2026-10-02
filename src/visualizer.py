import numpy as np
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR


def plot_predictions(
    y_test: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
) -> None:
    """
    Plot actual versus predicted target values.
    """

    y_test = np.asarray(y_test)
    predictions = np.asarray(predictions)

    # Calculate standard deviation of actual test targets
    std_y = np.std(y_test)

    # Determine the plotting range
    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    # Add padding around the plotting range
    padding = 0.05 * (maximum - minimum)

    axis_min = minimum - padding
    axis_max = maximum + padding

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot predictions
    ax.scatter(
        y_test,
        predictions,
        s=12,
        alpha=0.3,
        edgecolors="none",
        label="Predictions",
        rasterized=True,
    )

    # Ideal prediction line
    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.5,
        label="Ideal: y = x",
        color = "red"
    )

    # +1 standard deviation line
    ax.plot(
        [minimum, maximum],
        [minimum + std_y, maximum + std_y],
        linestyle="--",
        linewidth=1.5,
        label="+1 Std. Dev.",
        color = "green"
    )

    # -1 standard deviation line
    ax.plot(
        [minimum, maximum],
        [minimum - std_y, maximum - std_y],
        linestyle="--",
        linewidth=1.5,
        label="-1 Std. Dev.",
        color = "orange"
    )

    # Labels and title
    ax.set_xlabel("Actual Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title(f"{model_name}: Actual vs Predicted")

    # Add space around the reference lines
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.legend()

    # Adjust layout and save
    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png"
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)