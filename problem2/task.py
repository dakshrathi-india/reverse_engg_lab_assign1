import numpy as np
import matplotlib.pyplot as plt


class RE:
    def __init__(self, model):
        # Stores the black-box model for use in all functions.
        self.model = model

    def class1_prob(self, X):
        # Returns the probability of class 1 for all input points.
        X = np.asarray(X, dtype=float)
        return np.asarray(self.model.predict_proba(X), dtype=float)[:, 1]

    def prob_to_logit(self, prob):
        # Converts probabilities into logit values.
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return np.log(prob / (1 - prob))

    def sigmoid(self, X):
        # Converts logit values into probabilities.
        X = np.asarray(X, dtype=float)
        return 1 / (1 + np.exp(-X))

    # task1
    def plot_decision_boundary(self, filename="decision_boundary.png"):
        # Finds the p(class 1) = 0.5 boundary and saves its plot.
        x1 = np.linspace(-5, 5, 200)
        x2 = np.linspace(-5, 5, 200)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.reshape(-1), X2.reshape(-1)])

        pred_class = np.asarray(self.model.predict(X), dtype=int)

        # Contains the probability of class 1.
        pred_prob = self.class1_prob(X)

        class_grid = pred_class.reshape(X1.shape)
        prob_grid = pred_prob.reshape(X1.shape)

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.contourf(X1, X2, class_grid, levels=[-0.5, 0.5, 1.5], alpha=0.2)
        p = ax.scatter(X[:, 0], X[:, 1], c=pred_prob, s=5)
        ax.contour(X1, X2, prob_grid, levels=[0.5], colors="red", linewidths=2)
        ax.set_title("Model Prediction")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        fig.colorbar(p, ax=ax, label="Probability of Class 1")
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")

        return fig, ax

    # task2
    def recover_coefficients(self):
        # Recovers w1, w2 and b and returns probability MSE and boundary accuracy.
        X = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        )
        p = self.class1_prob(X)
        logit = self.prob_to_logit(p)

        # Model: z = w1*x1 + w2*x2 + b.
        intercept = logit[0]
        w1 = logit[1] - intercept
        w2 = logit[2] - intercept
        coeff = np.array([w1, w2])

        x1 = np.linspace(-5, 5, 200)
        x2 = np.linspace(-5, 5, 200)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.reshape(-1), X2.reshape(-1)])

        actual_prob = self.class1_prob(X)
        recover_logit = intercept + X @ coeff
        recover_prob = self.sigmoid(recover_logit)
        prob_mse = np.mean((actual_prob - recover_prob) ** 2)

        actual_class = np.asarray(self.model.predict(X), dtype=int).reshape(-1)
        recover_class = (recover_prob >= 0.5).astype(int)
        bound_acc = np.mean(actual_class == recover_class)

        return {
            "coefficients": coeff,
            "intercept": float(intercept),
            "probability_mse": float(prob_mse),
            "boundary_accuracy": float(bound_acc),
        }

    # task3
    def compute_feature_importance(
        self,
        feature_range=((-5, 5), (-5, 5)),
        sample=500,
        perturb_fraction=0.05,
        seed=42,
    ):
        # Perturbs one feature at a time and returns the probability changes.
        rng = np.random.default_rng(seed)
        X = np.column_stack(
            [rng.uniform(low, high, sample) for low, high in feature_range]
        )

        actual_prob = self.class1_prob(X)
        importance = np.zeros(2)
        perturb = np.zeros(2)

        for feature in range(2):
            low, high = feature_range[feature]
            delta = perturb_fraction * (high - low)
            perturb[feature] = delta

            perturb_X = np.copy(X)
            perturb_X[:, feature] += delta

            perturb_prob = self.class1_prob(perturb_X)
            importance[feature] = np.mean(np.abs(perturb_prob - actual_prob))

        total = np.sum(importance)
        if total > 0:
            normalized_importance = importance / total
        else:
            normalized_importance = importance

        most_important = np.argmax(normalized_importance) + 1

        return {
            "raw_importance": importance,
            "normalized_importance": normalized_importance,
            "perturbation": perturb,
            "most_important_feature": int(most_important),
        }

    def plot_feature_importance(self, filename="feature_importance.png"):
        # Plots and saves the normalized importance of both features.
        result = self.compute_feature_importance()
        importance = result["normalized_importance"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(
            ["Feature1", "Feature2"],
            importance,
            color=["steelblue", "darkorange"],
        )
        ax.set_title("Perturbation Based Feature Importance")
        ax.set_ylabel("Normalized Importance")
        ax.set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        return fig, ax

    def interaction_detection(self, sample=100, step=0.5, tolerance=1e-6, seed=42):
        # Uses mixed logit differences to detect interactions between x1 and x2.
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, size=(sample, 2))

        e1 = np.array([step, 0])
        e2 = np.array([0, step])

        test_cases = np.array(
            [
                X + e1 + e2,
                X + e1,
                X + e2,
                X,
            ]
        )

        interaction_scores = np.zeros(sample)
        for i in range(4):
            p = self.class1_prob(test_cases[i])
            logit = self.prob_to_logit(p)

            if i == 1 or i == 2:
                interaction_scores -= logit
            else:
                interaction_scores += logit

        max_interaction = np.max(np.abs(interaction_scores))

        return {
            "max_interaction": float(max_interaction),
            "mean_interaction": float(np.mean(np.abs(interaction_scores))),
            "interaction_detected": bool(max_interaction > tolerance),
        }

    # task4
    def test_robustness(self, grid_limit=5, grid_size=200, number_of_points=100):
        # Perturbs points near the boundary and returns class-flip statistics.
        recover = self.recover_coefficients()
        coeff = recover["coefficients"]
        intercept = recover["intercept"]

        x1 = np.linspace(-grid_limit, grid_limit, grid_size)
        x2 = np.linspace(-grid_limit, grid_limit, grid_size)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.reshape(-1), X2.reshape(-1)])

        actual_prob = self.class1_prob(X)
        indices = np.argsort(np.abs(actual_prob - 0.5))[:number_of_points]

        boundary_points = X[indices]
        boundary_prob = actual_prob[indices]
        logits = intercept + boundary_points @ coeff

        coeff_norm = np.linalg.norm(coeff)
        directions = -np.sign(logits)[:, None] * coeff[None, :] / coeff_norm
        distances = np.abs(logits) / coeff_norm
        adversarial_points = boundary_points + (distances + 1e-3)[:, None] * directions

        original_classes = np.asarray(self.model.predict(boundary_points), dtype=int)
        adversarial_classes = np.asarray(
            self.model.predict(adversarial_points), dtype=int
        )
        class_flips = original_classes != adversarial_classes
        perturbation_sizes = np.linalg.norm(
            adversarial_points - boundary_points, axis=1
        )

        return {
            "tested_points": number_of_points,
            "class_flips": int(np.sum(class_flips)),
            "flip_rate": float(np.mean(class_flips)),
            "mean_perturbation": float(np.mean(perturbation_sizes)),
            "mean_boundary_confidence": float(np.mean(np.abs(boundary_prob - 0.5))),
        }

    def plot_confidence_heatmap(
        self,
        grid_limit=5,
        grid_size=200,
        filename="confidence_heatmap.png",
    ):
        # Plots and saves prediction confidence over the input grid.
        x1 = np.linspace(-grid_limit, grid_limit, grid_size)
        x2 = np.linspace(-grid_limit, grid_limit, grid_size)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.reshape(-1), X2.reshape(-1)])

        actual_prob = self.class1_prob(X)
        confidence = np.maximum(actual_prob, 1 - actual_prob)

        prob_grid = actual_prob.reshape(X1.shape)
        confidence_grid = confidence.reshape(X1.shape)

        fig, ax = plt.subplots(figsize=(7, 6))
        # heatmap will be used in the colorbar
        heatmap = ax.contourf(
            X1,
            X2,
            confidence_grid,
            levels=np.linspace(0.5, 1, 21),
            cmap="viridis",
        )
        ax.contour(X1, X2, prob_grid, levels=[0.5], colors="red", linewidths=2)
        ax.set_title("Model Confidence Heatmap")
        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")

        fig.colorbar(heatmap, ax=ax, label="Prediction confidence")
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")

        return fig, ax

    def plot_coefficient_recovery(
        self,
        true_coeff,
        filename="coefficient_recovery.png",
    ):
        # Compares true and recovered coefficients and saves their plot.
        recover = self.recover_coefficients()
        est_coeff = recover["coefficients"]

        fig, ax = plt.subplots(figsize=(7, 6))
        positions = np.arange(2)
        labels = ["Feature1", "Feature2"]
        true_coeff = np.asarray(true_coeff, dtype=float).reshape(-1)

        coeff_error = np.linalg.norm(est_coeff - true_coeff)
        width = 0.35

        ax.bar(
            positions - width / 2,
            true_coeff,
            width,
            label="True Coefficients",
        )
        ax.bar(
            positions + width / 2,
            est_coeff,
            width,
            label="Recovered Coefficients",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Coefficient value")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("True vs Recovered Coefficients")
        ax.legend()

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")

        return fig, ax, float(coeff_error)

    def evaluate_recovery(self, true_coeff, samples=1000, seed=42):
        # Returns coefficient recovery error and decision-boundary accuracy.
        rng = np.random.default_rng(seed)
        X = rng.uniform(-5, 5, size=(samples, 2))

        recover = self.recover_coefficients()
        recover_coeff = recover["coefficients"]
        recover_intercept = recover["intercept"]

        true_coeff = np.asarray(true_coeff, dtype=float).reshape(-1)
        coeff_error = np.linalg.norm(true_coeff - recover_coeff)
        actual_class = np.asarray(self.model.predict(X), dtype=int)

        z = recover_intercept + X @ recover_coeff
        recover_prob = self.sigmoid(z)
        recover_class = (recover_prob >= 0.5).astype(int)
        boundary_accuracy = np.mean(actual_class == recover_class)

        return float(coeff_error), float(boundary_accuracy)
