import numpy as np
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


class RE:
    def __init__(self, model):
        self.model = model

    def get_probability(self, X):
        prediction = self.model.predict(X, verbose=0)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def predict_class(self, X):
        return (self.get_probability(X) >= 0.5).astype(int)

    def get_layer_output(self, X, layer_id):
        dense_layers = [layer for layer in self.model.layers if hasattr(layer, "units")]
        layer = dense_layers[layer_id - 1]

        activation_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=layer.output,
        )
        return np.asarray(activation_model.predict(X, verbose=0), dtype=float)

    # Task 1: input-feature importance
    def compute_input_importance(self, X, seed=42):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(seed)
        actual_prob = self.get_probability(X)
        importance = np.zeros(X.shape[1])

        for feature in range(X.shape[1]):
            perturbed_X = X.copy()
            perturbed_X[:, feature] = rng.permutation(perturbed_X[:, feature])
            perturbed_prob = self.get_probability(perturbed_X)
            importance[feature] = np.mean(np.abs(perturbed_prob - actual_prob))

        normalized_importance = importance / np.sum(importance)
        ranking = np.argsort(-importance)

        return {
            "raw_importance": importance,
            "normalized_importance": normalized_importance,
            "ranking": ranking,
            "top_10_features": ranking[:10] + 1,
        }

    # Tasks 2 and 3: hidden-layer analysis
    def analyze_layer(self, X, y, layer_id, redundancy_limit=0.95):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        activations = self.get_layer_output(X, layer_id)

        mean_activation = np.mean(activations, axis=0)
        std_activation = np.std(activations, axis=0)
        zero_fraction = np.mean(np.abs(activations) < 1e-10, axis=0)
        class_0_mean = np.mean(activations[y == 0], axis=0)
        class_1_mean = np.mean(activations[y == 1], axis=0)
        class_separation = np.abs(class_1_mean - class_0_mean) / (
            std_activation + 1e-10
        )

        neuron_correlation = np.nan_to_num(np.corrcoef(activations, rowvar=False))
        number_of_neurons = activations.shape[1]
        redundant_pairs = [
            (i + 1, j + 1)
            for i in range(number_of_neurons)
            for j in range(i + 1, number_of_neurons)
            if abs(neuron_correlation[i, j]) >= redundancy_limit
        ]

        X_centered = X - np.mean(X, axis=0)
        A_centered = activations - np.mean(activations, axis=0)
        denominator = np.sqrt(
            np.sum(X_centered**2, axis=0)[:, None]
            * np.sum(A_centered**2, axis=0)[None, :]
        )
        input_correlation = np.divide(
            X_centered.T @ A_centered,
            denominator,
            out=np.zeros((X.shape[1], number_of_neurons)),
            where=denominator > 0,
        )

        top_input_features = np.argmax(np.abs(input_correlation), axis=0) + 1
        cluster_score = float(silhouette_score(activations, y))

        return {
            "layer_id": layer_id,
            "activations": activations,
            "labels": y,
            "mean_activation": mean_activation,
            "std_activation": std_activation,
            "zero_fraction": zero_fraction,
            "class_0_mean": class_0_mean,
            "class_1_mean": class_1_mean,
            "class_separation": class_separation,
            "neuron_correlation": neuron_correlation,
            "redundant_pairs": redundant_pairs,
            "dead_neurons": np.where(zero_fraction >= 0.95)[0] + 1,
            "input_correlation": input_correlation,
            "top_input_features": top_input_features,
            "silhouette_score": cluster_score,
        }

    def compute_tsne_projection(self, layer_analysis, max_samples=1000, seed=42):
        activations = layer_analysis["activations"]
        labels = layer_analysis["labels"]
        rng = np.random.default_rng(seed)

        if len(activations) > max_samples:
            indices = rng.choice(len(activations), max_samples, replace=False)
            activations = activations[indices]
            labels = labels[indices]

        projection = TSNE(
            n_components=2,
            perplexity=min(30, len(activations) - 1),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(activations)

        return {"projection": projection, "labels": labels}

    # Task 4: pairwise input-feature interactions
    def detect_feature_interactions(self, X, samples=300, seed=42):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(seed)

        if len(X) > samples:
            X = X[rng.choice(len(X), samples, replace=False)]

        actual_prob = self.get_probability(X)
        number_of_features = X.shape[1]
        interaction_matrix = np.zeros((number_of_features, number_of_features))

        permutations = [rng.permutation(len(X)) for _ in range(number_of_features)]
        single_perturb_prob = []

        for feature in range(number_of_features):
            perturbed_X = X.copy()
            perturbed_X[:, feature] = X[permutations[feature], feature]
            single_perturb_prob.append(self.get_probability(perturbed_X))

        for i in range(number_of_features):
            for j in range(i + 1, number_of_features):
                perturbed_X = X.copy()
                perturbed_X[:, i] = X[permutations[i], i]
                perturbed_X[:, j] = X[permutations[j], j]
                pair_prob = self.get_probability(perturbed_X)

                score = np.mean(
                    np.abs(
                        pair_prob
                        - single_perturb_prob[i]
                        - single_perturb_prob[j]
                        + actual_prob
                    )
                )
                interaction_matrix[i, j] = score
                interaction_matrix[j, i] = score

        pairs = [
            (i + 1, j + 1, interaction_matrix[i, j])
            for i in range(number_of_features)
            for j in range(i + 1, number_of_features)
        ]
        pairs.sort(key=lambda pair: pair[2], reverse=True)

        return {
            "interaction_matrix": interaction_matrix,
            "top_interactions": pairs[:10],
            "maximum_interaction": float(pairs[0][2]),
        }

    def compute_neuron_contributions(self, layer_analysis):
        contribution = np.abs(
            layer_analysis["class_1_mean"] - layer_analysis["class_0_mean"]
        )
        normalized_contribution = contribution / np.sum(contribution)

        return {
            "layer_id": layer_analysis["layer_id"],
            "raw_contribution": contribution,
            "normalized_contribution": normalized_contribution,
            "ranking": np.argsort(-contribution) + 1,
        }

    def layer_pruning_analysis(self, y, layer_analysis):
        y = np.asarray(y, dtype=int).reshape(-1)
        layer_id = layer_analysis["layer_id"]
        activations = layer_analysis["activations"]
        contribution = self.compute_neuron_contributions(layer_analysis)
        removal_order = np.argsort(contribution["raw_contribution"])

        dense_layers = [layer for layer in self.model.layers if hasattr(layer, "units")]
        required_layer = dense_layers[layer_id - 1]
        model_layer_index = self.model.layers.index(required_layer)

        tail_input = tf.keras.Input(shape=(activations.shape[1],))
        output = tail_input
        for layer in self.model.layers[model_layer_index + 1 :]:
            output = layer(output, training=False)
        tail_model = tf.keras.Model(inputs=tail_input, outputs=output)

        number_of_neurons = activations.shape[1]
        retained_neurons = np.arange(number_of_neurons, -1, -1)
        accuracy = []

        for retained in retained_neurons:
            removed = number_of_neurons - retained
            mask = np.ones(number_of_neurons)
            mask[removal_order[:removed]] = 0

            pruned_prob = np.asarray(
                tail_model.predict(activations * mask, verbose=0), dtype=float
            ).reshape(-1)
            pruned_class = (pruned_prob >= 0.5).astype(int)
            accuracy.append(np.mean(pruned_class == y))

        return {
            "layer_id": layer_id,
            "retained_neurons": retained_neurons,
            "accuracy": np.asarray(accuracy),
            "removal_order": removal_order + 1,
        }

    # Plotting functions: ax is always supplied by main.py
    def plot_feature_importance_bar(self, input_importance, ax):
        ranking = input_importance["ranking"][:10]
        values = input_importance["normalized_importance"][ranking]
        labels = [f"F{feature + 1}" for feature in ranking]

        ax.bar(labels, values, color="steelblue")
        ax.set_title("Top-10 Input Features")
        ax.set_xlabel("Input feature")
        ax.set_ylabel("Normalized importance")
        ax.tick_params(axis="x", rotation=45)

    def plot_layer_activations(self, layer_analysis, ax, max_samples=100):
        activations = layer_analysis["activations"]
        labels = layer_analysis["labels"]
        half = max_samples // 2
        indices_0 = np.where(labels == 0)[0][:half]
        indices_1 = np.where(labels == 1)[0][:half]
        indices = np.concatenate([indices_0, indices_1])

        heatmap = ax.imshow(activations[indices], aspect="auto", cmap="viridis")
        ax.axhline(len(indices_0) - 0.5, color="white", linewidth=1.5)
        ax.set_title(f"Hidden Layer {layer_analysis['layer_id']} Activations")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Samples: class 0 then class 1")
        ax.set_xticks(np.arange(activations.shape[1]))
        ax.set_xticklabels(np.arange(1, activations.shape[1] + 1))
        ax.figure.colorbar(heatmap, ax=ax, label="Activation")

    def plot_hidden_layer_tsne(self, tsne_result, ax):
        projection = tsne_result["projection"]
        labels = tsne_result["labels"]

        points = ax.scatter(
            projection[:, 0], projection[:, 1], c=labels, cmap="coolwarm", s=12
        )
        ax.set_title("Second Hidden Layer t-SNE")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.figure.colorbar(points, ax=ax, ticks=[0, 1], label="Class")

    def plot_interaction_matrix(self, interactions, ax):
        matrix = interactions["interaction_matrix"]
        heatmap = ax.imshow(matrix, cmap="magma")

        ax.set_title("Feature Interaction Matrix")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Feature")
        ax.set_xticks(np.arange(matrix.shape[0]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels(np.arange(1, matrix.shape[0] + 1), fontsize=6)
        ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1), fontsize=6)
        ax.figure.colorbar(heatmap, ax=ax, label="Interaction score")

    def plot_neuron_contributions(self, contributions, ax):
        values = contributions["normalized_contribution"]
        positions = np.arange(1, len(values) + 1)

        ax.bar(positions, values, color="darkorange")
        ax.set_title(f"Layer {contributions['layer_id']} Neuron Contributions")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Normalized contribution")
        ax.set_xticks(positions)

    def plot_layer_pruning_analysis(self, pruning_result, ax):
        ax.plot(
            pruning_result["retained_neurons"],
            pruning_result["accuracy"],
            marker="o",
            color="green",
        )
        ax.set_title(f"Layer {pruning_result['layer_id']} Pruning Analysis")
        ax.set_xlabel("Number of retained neurons")
        ax.set_ylabel("Classification accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
