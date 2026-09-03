import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from task import RE

MODEL_FILE = "neural_network_mnist_20260901_210852_model.h5"
PREPROCESSOR_FILE = "neural_network_mnist_20260901_210852_preprocessing.joblib"


def apply_preprocessor(preprocessor, X):
    # Applies either a direct transformer or transformers stored in a dictionary.
    if hasattr(preprocessor, "transform"):
        transformed = preprocessor.transform(X)
    elif isinstance(preprocessor, dict):
        transformed = None

        for key in ("preprocessor", "transformer", "pipeline"):
            if key in preprocessor and hasattr(preprocessor[key], "transform"):
                transformed = preprocessor[key].transform(X)
                break

        if transformed is None:
            transformed = X
            used_transformer = False
            for key in ("scaler", "pca"):
                if key in preprocessor and hasattr(preprocessor[key], "transform"):
                    transformed = preprocessor[key].transform(transformed)
                    used_transformer = True

            if not used_transformer:
                raise ValueError(
                    "No usable transformer was found in the preprocessor file"
                )
    else:
        raise TypeError("The supplied preprocessor does not provide transform()")

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed, dtype=np.float32)


def prepare_mnist(model, preprocessor):
    # Finds the correct pixel scaling and zero/non-zero output-label mapping.
    (_, _), (images, digit_labels) = tf.keras.datasets.mnist.load_data()
    flat_images = images.reshape(len(images), -1).astype(np.float32)

    best_result = None
    candidates = {
        "normalized [0,1]": flat_images / 255.0,
        "raw [0,255]": flat_images,
    }

    for scale_name, candidate in candidates.items():
        try:
            processed = apply_preprocessor(preprocessor, candidate)
            probability = np.asarray(
                model.predict(processed, verbose=0), dtype=float
            ).reshape(-1)
            predicted_class = (probability >= 0.5).astype(int)

            nonzero_labels = (digit_labels != 0).astype(int)
            zero_labels = (digit_labels == 0).astype(int)
            nonzero_accuracy = np.mean(predicted_class == nonzero_labels)
            zero_accuracy = np.mean(predicted_class == zero_labels)

            if nonzero_accuracy >= zero_accuracy:
                binary_labels = nonzero_labels
                mapping = "class 1 = non-zero, class 0 = zero"
                accuracy = nonzero_accuracy
            else:
                binary_labels = zero_labels
                mapping = "class 1 = zero, class 0 = non-zero"
                accuracy = zero_accuracy

            if best_result is None or accuracy > best_result[0]:
                best_result = (
                    accuracy,
                    processed,
                    binary_labels,
                    digit_labels,
                    scale_name,
                    mapping,
                )
        except (TypeError, ValueError):
            continue

    if best_result is None:
        raise ValueError("The MNIST data could not be transformed by the preprocessor")

    return best_result


def create_balanced_sample(X, binary_labels, digit_labels, each_class=500, seed=42):
    # Returns an equal number of zero and non-zero images for fair analysis.
    rng = np.random.default_rng(seed)
    zero_indices = np.where(digit_labels == 0)[0]
    nonzero_indices = np.where(digit_labels != 0)[0]

    count = min(each_class, len(zero_indices), len(nonzero_indices))
    indices = np.concatenate(
        [
            rng.choice(zero_indices, count, replace=False),
            rng.choice(nonzero_indices, count, replace=False),
        ]
    )
    rng.shuffle(indices)

    return X[indices], binary_labels[indices]


def main():
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    result = prepare_mnist(model, preprocessor)
    accuracy, X, y, digit_labels, scale_name, mapping = result
    X_analysis, y_analysis = create_balanced_sample(X, y, digit_labels)

    print("Selected input scaling:", scale_name)
    print("Detected label mapping:", mapping)
    print("Model accuracy using this mapping:", float(accuracy))
    print("Processed input shape:", X.shape)
    print("Analysis sample shape:", X_analysis.shape)
    print()

    re = RE(model)

    # Task 1: input-feature importance.
    input_importance = re.compute_input_importance(X_analysis)
    print("Top-10 input features:", input_importance["top_10_features"])
    print(
        "Top-10 importance values:",
        input_importance["normalized_importance"][input_importance["ranking"][:10]],
    )

    # Tasks 2 and 3: hidden-layer activation analysis.
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

    # Task 4: pairwise input-feature interactions.
    interactions = re.detect_feature_interactions(X_analysis)
    print("Maximum interaction:", interactions["maximum_interaction"])
    print("Top feature interactions:", interactions["top_interactions"])

    # Required neuron-contribution and pruning experiments use hidden layer 2.
    contributions = re.compute_neuron_contributions(second_layer)
    pruning = re.layer_pruning_analysis(X_analysis, y_analysis, second_layer)
    print("Second-layer neuron ranking:", contributions["ranking"])
    print("Pruning order:", pruning["removal_order"])
    print("Retained neurons:", pruning["retained_neurons"])
    print("Accuracy after pruning:", pruning["accuracy"])

    # Save all six plots individually.
    figures = []
    figures.append(re.plot_feature_importance_bar(input_importance)[0])
    figures.append(re.plot_layer_activations(first_layer)[0])
    figures.append(re.plot_hidden_layer_tsne(tsne_result)[0])
    figures.append(re.plot_interaction_matrix(interactions)[0])
    figures.append(re.plot_neuron_contributions(contributions)[0])
    figures.append(re.plot_layer_pruning_analysis(pruning)[0])

    for fig in figures:
        plt.close(fig)

    # Save and display the combined six-panel assignment figure.
    re.plot_network_analysis(
        input_importance,
        first_layer,
        tsne_result,
        interactions,
        contributions,
        pruning,
    )
    plt.show()


if __name__ == "__main__":
    main()
