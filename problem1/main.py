import joblib
import matplotlib.pyplot as plt
from task import RE


def main():
    model = [
        joblib.load("concrete_strength_20260901_164159_model.joblib"),
        joblib.load("boston_housing_20260901_164311_model.joblib"),
    ]

    for i in range(2):
        if i == 0:
            print("For concrete strength model")
        else:
            print("For boston housing model")
        print()

        obj = RE(model[i])
        intercept = obj.recover_intercept()
        coefficients = obj.recover_coefficients()

        error_result = obj.estimate_model_error()
        mse_estimate = error_result["recover_mse"]

        nonlinearity_result = obj.detect_nonlinearity()

        print("Recovered intercept:", intercept)
        print("Recovered coefficients:", coefficients)

        print("Recovery MSE:", mse_estimate)
        print("Actual Prediction variance:", error_result["actual_prediction_var"])
        print("Recovered Prediction variance:", error_result["recover_prediction_var"])
        print("Non-linearity results:", nonlinearity_result)
        if i == 0:
            obj.plot_model_behaviour("concrete_strength_behaviour.png")
        else:
            obj.plot_model_behaviour("boston_housing_behaviour.png")
        print()

        plt.show()


if __name__ == "__main__":
    main()
