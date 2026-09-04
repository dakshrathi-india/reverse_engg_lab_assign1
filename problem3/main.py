import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from task import RE

MODEL_FILE = "neural_network_mnist_20260901_210852_model.h5"
PREPROCESSOR_FILE = "neural_network_mnist_20260901_210852_preprocessing.joblib"


def apply_preprocessor(preprocessor, X):
    # Confirmed preprocessing order: scaler -> PCA.
    X = preprocessor["scaler"].transform(X)
    X = preprocessor["pca"].transform(X)
    return np.asarray(X, dtype=np.float32)


def prepare_mnist(model, preprocessor):
    # Confirmed setup: normalized pixels and class 1 = zero.
    (_, _), (images, digit_labels) = tf.keras.datasets.mnist.load_data()

    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    X = apply_preprocessor(preprocessor, X)

    y = (digit_labels == 0).astype(int)
    accuracy = np.mean((model.predict(X, verbose=0).reshape(-1) >= 0.5) == y)

    return X, y, digit_labels, accuracy


def create_balanced_sample(X, y, digit_labels, each_class=500, seed=42):
    rng = np.random.default_rng(seed)

    zero_indices = rng.choice(np.where(digit_labels == 0)[0], each_class, replace=False)
    nonzero_indices = rng.choice(
        np.where(digit_labels != 0)[0], each_class, replace=False
    )

    indices = np.concatenate([zero_indices, nonzero_indices])
    rng.shuffle(indices)

    return X[indices], y[indices]


def main():
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    X, y, digit_labels, accuracy = prepare_mnist(model, preprocessor)
    X_analysis, y_analysis = create_balanced_sample(X, y, digit_labels)

    print("Input scaling: normalized [0,1]")
    print("Preprocessing order: scaler -> PCA")
    print("Label mapping: class 1 = zero, class 0 = non-zero")
    print("Model accuracy:", float(accuracy))
    print("Processed input shape:", X.shape)
    print("Analysis sample shape:", X_analysis.shape)
    print()

    re = RE(model)

    # Task 1: input-feature importance
    input_importance = re.compute_input_importance(X_analysis)
    print("Top-10 input features:", input_importance["top_10_features"])
    print(
        "Top-10 importance values:",
        input_importance["normalized_importance"][input_importance["ranking"][:10]],
    )

    # Tasks 2 and 3: hidden-layer activation analysis
    first_layer = re.analyze_layer(X_analysis, y_analysis, layer_id=1)
    second_layer = re.analyze_layer(X_analysis, y_analysis, layer_id=2)
    tsne_result = re.compute_tsne_projection(second_layer)

    print("First-layer dead neurons:", first_layer["dead_neurons"])
    print("First-layer redundant pairs:", first_layer["redundant_pairs"])
    print(
        "First-layer top input feature per neuron:", first_layer["top_input_features"]
    )
    print("First-layer silhouette score:", first_layer["silhouette_score"])
    print("Second-layer dead neurons:", second_layer["dead_neurons"])
    print("Second-layer redundant pairs:", second_layer["redundant_pairs"])
    print("Second-layer silhouette score:", second_layer["silhouette_score"])

    # Task 4: pairwise input-feature interactions
    interactions = re.detect_feature_interactions(X_analysis)
    print("Maximum interaction:", interactions["maximum_interaction"])
    print("Top feature interactions:", interactions["top_interactions"])

    # Neuron contribution and pruning analysis on hidden layer 2
    contributions = re.compute_neuron_contributions(second_layer)
    pruning = re.layer_pruning_analysis(y_analysis, second_layer)

    print("Second-layer neuron ranking:", contributions["ranking"])
    print("Pruning order:", pruning["removal_order"])
    print("Retained neurons:", pruning["retained_neurons"])
    print("Accuracy after pruning:", pruning["accuracy"])

    # One combined figure only. All axes are created here and passed to task.py.
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    re.plot_feature_importance_bar(input_importance, axes[0, 0])
    re.plot_layer_activations(first_layer, axes[0, 1])
    re.plot_hidden_layer_tsne(tsne_result, axes[0, 2])
    re.plot_interaction_matrix(interactions, axes[1, 0])
    re.plot_neuron_contributions(contributions, axes[1, 1])
    re.plot_layer_pruning_analysis(pruning, axes[1, 2])

    fig.tight_layout()
    fig.savefig("network_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
