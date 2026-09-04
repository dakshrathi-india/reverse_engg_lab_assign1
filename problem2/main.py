import joblib
import matplotlib.pyplot as plt
import numpy as np

from task import RE


def main():
    model = joblib.load("breast_cancer_20260901_204528_model.joblib")
    re = RE(model["model"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Task 1: decision boundary plot
    re.plot_decision_boundary(ax=axes[0, 0])

    # Task 2: recovered coefficients and model-recovery errors
    recover = re.recover_coefficients()
    print("Recovered coefficients:", recover["coefficients"])
    print("Recovered intercept:", recover["intercept"])
    print("Probability MSE:", recover["probability_mse"])
    print("Decision Boundary Accuracy:", recover["boundary_accuracy"])

    # Task 3: feature importance and feature-interaction experiment
    importance = re.compute_feature_importance()
    print("Feature importance:", importance)
    re.plot_feature_importance(ax=axes[0, 1])

    interaction = re.interaction_detection()
    print("Interaction result:", interaction)

    # Task 4: confidence heatmap and adversarial robustness experiment
    robustness = re.test_robustness()
    print("Robustness result:", robustness)
    re.plot_confidence_heatmap(ax=axes[1, 0])

    true_coeff = np.asarray(model["model"].coef_, dtype=float).reshape(-1)
    re.plot_coefficient_recovery(true_coeff, ax=axes[1, 1])

    coefficient_error, boundary_accuracy = re.evaluate_recovery(true_coeff)
    print(f"Coefficient Recovery Error: {coefficient_error:.4f}")
    print(f"Decision Boundary Accuracy: {boundary_accuracy:.4f}")

    # Displays all figures created above in separate windows.
    plt.show()
    fig.savefig("problem2_combined", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
