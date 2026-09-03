import numpy as np
import matplotlib.pyplot as plt

# model : y_pred = w1x1 + w2x2 + w3x3 + b


class RE:
    def __init__(self, model):
        self.model = model

    # task1
    def recover_intercept(self):
        # return b
        X = np.zeros((1, 3))
        return float(self.model.predict(X)[0])

    # task2
    def recover_coefficients(self):
        # return [w1,w2,w3]
        X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pred = np.asarray(self.model.predict(X)).reshape(-1)
        coeff = pred[1:] - pred[0]
        return coeff

    # task3
    def estimate_model_error(self, samples=100, seed=42):
        # return the coefficient-recovery MSE
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(samples, 3))

        actual_pred = np.asarray(self.model.predict(X)).reshape(-1)
        b = self.recover_intercept()
        coeff = self.recover_coefficients()

        recover_pred = b + X @ coeff
        recover_mse = np.mean((actual_pred - recover_pred) ** 2)
        actual_pred_var = np.var(actual_pred)
        recover_pred_var = np.var(recover_pred)

        return {
            "recover_mse": float(recover_mse),
            "actual_prediction_var": float(actual_pred_var),
            "recover_prediction_var": float(recover_pred_var),
        }

    # task4
    def detect_nonlinearity(self, tolerance=1e-6):
        def predict_one(x):
            X = np.array([x], dtype=float).reshape(1, -1)
            return float(np.asarray(self.model.predict(X))[0])

        coeff = self.recover_coefficients()
        intercept = self.recover_intercept()

        # test1 (to find non linear features)
        # checking if there exist some term like w_i*(x_i^a) such that a is not equal 1
        nonlinear_feature = []
        for i in range(3):
            X = np.zeros((1, 3))
            X[0, i] = 1
            Y = 2 * X
            diff = predict_one(Y) - predict_one(X) - coeff[i]
            if abs(diff) > tolerance:
                nonlinear_feature.append(i + 1)

        # test2 (to find interactions between different features)
        # check if there exist some term as a combination of 2 features like k*(x_i^a)**(x_j^b)
        interaction = []
        for i in range(3):
            for j in range(i + 1, 3):
                Xi = np.zeros((1, 3))
                Xj = np.zeros((1, 3))
                Xi[0, i] = 1
                Xj[0, j] = 1
                diff = (
                    predict_one(Xi + Xj)
                    - (predict_one(Xi) + predict_one(Xj))
                    + intercept
                )
                if abs(diff) > tolerance:
                    interaction.append((i + 1, j + 1))

        return {
            "nonlinear_feature": nonlinear_feature,
            "interactions": interaction,
            "nonlinear_model": bool(nonlinear_feature or interaction),
        }

    # bonus
    def plot_model_behaviour(self, filename="model_behaviour.png"):
        # in this we will be comparing the 3d scatter plot of the actual model and the recovered model
        x1 = np.linspace(-1, 1, 10)
        x2 = np.linspace(-1, 1, 10)
        x3 = np.linspace(-1, 1, 10)

        X1, X2, X3 = np.meshgrid(x1, x2, x3)
        X = np.column_stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)])

        actual_pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        recovered_pred = self.recover_intercept() + X @ self.recover_coefficients()

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})
        for i in range(2):
            if i == 0:
                l = "Original Model"
                t = actual_pred
            else:
                l = "Recovered Model"
                t = recovered_pred

            p = ax[i].scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
            ax[i].set_title(l)
            ax[i].set_xlabel("x1")
            ax[i].set_ylabel("x2")
            ax[i].set_zlabel("x3")

            fig.colorbar(p, ax=ax[i], label="Prediciton")
            fig.tight_layout()
            fig.savefig(filename, dpi=150, bbox_inches="tight")

        return fig, ax
