import pandas as pd
import numpy as np

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def load_data(path, features, label_col):
    df = pd.read_csv(path)
    df = df[features +[label_col]].dropna()

    features_matrix = df[features].to_numpy()
    features_matrix = (features_matrix - features_matrix.mean(axis=0)) / features_matrix.std(axis=0)

    y_raw = df[label_col].to_numpy()
    labels_onehot = np.zeros((len(y_raw), len(houses)))
    for i, h in enumerate(houses):
        labels_onehot[:, i] = (y_raw == h).astype(int)

    return features_matrix, labels_onehot


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def train_logistic_regression(features_matrix, labels_onehot, lr=0.1, epochs=1000):
    num_samples, num_features = features_matrix.shape
    num_classes = labels_onehot.shape[1]

    weights = np.zeros((num_features, num_classes))
    biases = np.zeros((1, num_classes))

    for epoch in range(epochs):
        z = np.dot(features_matrix, weights) + biases
        y_pred = softmax(z)

        dz = y_pred - labels_onehot
        dW = np.dot(features_matrix.T, dz) / num_samples
        db = np.sum(dz, axis=0, keepdims=True) / num_samples

        weights -= lr * dW
        biases -= lr * db

        if epoch % 100 == 0:
            loss = cross_entropy(labels_onehot, y_pred)
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return weights, biases


def save_training(weights, biases):
    with open("./model.txt", "w") as f:
        f.write(str(weights) + "|" + str(biases))


def main():
    print("train")
    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    features_matrix, labels_onehot = load_data("./datasets/dataset_train.csv", features, "Hogwarts House")
    weights, biases = train_logistic_regression(features_matrix, labels_onehot, lr=0.01, epochs=1000)
    save_training(weights, biases)


if __name__ == "__main__":
    main()