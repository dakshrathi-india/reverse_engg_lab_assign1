import joblib
import numpy as np
import tensorflow as tf

MODEL_FILE = "neural_network_mnist_20260901_210852_model.h5"
PREPROCESSOR_FILE = "neural_network_mnist_20260901_210852_preprocessing.joblib"


model = tf.keras.models.load_model(MODEL_FILE, compile=False)

preprocessor = joblib.load(PREPROCESSOR_FILE)

scaler = preprocessor["scaler"]
pca = preprocessor["pca"]


# 1. Check preprocessing dimensions and order
print("\n--- PREPROCESSOR ---")
print("Preprocessor type:", type(preprocessor))
print("Preprocessor keys:", preprocessor.keys())
print("Scaler input features:", scaler.n_features_in_)
print("PCA input features:", pca.n_features_in_)
print("PCA output features:", pca.n_components_)


# 2. Determine whether training used raw or normalized pixels
if scaler.n_features_in_ == 784:
    training_pixel_mean = scaler.mean_
else:
    training_pixel_mean = pca.mean_

print("\n--- PIXEL SCALE ---")
print("Minimum training mean:", np.min(training_pixel_mean))
print("Maximum training mean:", np.max(training_pixel_mean))
print("Average training mean:", np.mean(training_pixel_mean))

if np.max(training_pixel_mean) <= 1:
    normalized = True
    print("Detected scaling: normalized [0, 1]")
else:
    normalized = False
    print("Detected scaling: raw [0, 255]")


# 3. Load and flatten MNIST
(_, _), (images, digit_labels) = tf.keras.datasets.mnist.load_data()

X = images.reshape(len(images), -1).astype(np.float32)

if normalized:
    X = X / 255.0


# 4. Apply preprocessing in the correct order
if scaler.n_features_in_ == 784:
    print("Detected order: scaler -> PCA")
    X = scaler.transform(X)
    X = pca.transform(X)
else:
    print("Detected order: PCA -> scaler")
    X = pca.transform(X)
    X = scaler.transform(X)

print("Processed data shape:", X.shape)


# 5. Check the output-label mapping
probability = model.predict(X, verbose=0).reshape(-1)

predicted_class = (probability >= 0.5).astype(int)

labels_zero_positive = (digit_labels == 0).astype(int)

labels_nonzero_positive = (digit_labels != 0).astype(int)

zero_positive_accuracy = np.mean(predicted_class == labels_zero_positive)

nonzero_positive_accuracy = np.mean(predicted_class == labels_nonzero_positive)

print("\n--- LABEL MAPPING ---")
print("Accuracy when class 1 = zero:", zero_positive_accuracy)

print("Accuracy when class 1 = non-zero:", nonzero_positive_accuracy)

print("Minimum predicted probability:", np.min(probability))
print("Maximum predicted probability:", np.max(probability))
print("Average predicted probability:", np.mean(probability))


# 6. Check the exact neural-network architecture
print("\n--- MODEL ARCHITECTURE ---")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

for index, layer in enumerate(model.layers):
    print(
        "Layer index:",
        index,
        "| name:",
        layer.name,
        "| type:",
        layer.__class__.__name__,
        "| units:",
        getattr(layer, "units", None),
        "| activation:",
        getattr(getattr(layer, "activation", None), "__name__", None),
    )


# 7. Check hidden-layer output shapes
dense_layers = [layer for layer in model.layers if hasattr(layer, "units")]

first_layer_model = tf.keras.Model(inputs=model.inputs, outputs=dense_layers[0].output)

second_layer_model = tf.keras.Model(inputs=model.inputs, outputs=dense_layers[1].output)

first_layer_output = first_layer_model.predict(X[:10], verbose=0)

second_layer_output = second_layer_model.predict(X[:10], verbose=0)

print("\n--- HIDDEN-LAYER OUTPUTS ---")
print("First hidden-layer shape:", first_layer_output.shape)

print("Second hidden-layer shape:", second_layer_output.shape)
