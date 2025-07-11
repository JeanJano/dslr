import pandas as pd
import numpy as np
import sys

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def load_data(path, features, label_col):
    try:
        df = pd.read_csv(path)
        df = df[features +[label_col]].dropna()

        features_matrix = df[features].to_numpy()
        features_matrix = (features_matrix - features_matrix.mean(axis=0)) / features_matrix.std(axis=0)

        y_raw = df[label_col].to_numpy()
        labels_onehot = np.zeros((len(y_raw), len(houses)))
        for i, h in enumerate(houses):
            labels_onehot[:, i] = (y_raw == h).astype(int)

        return features_matrix, labels_onehot
    except FileNotFoundError:
        print(f"file {path} not found")
        sys.exit(1)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def train_logistic_regression(features_matrix, labels_onehot, lr=0.1, epochs=1000):
    num_samples, num_features = features_matrix.shape
    num_classes = labels_onehot.shape[1]

    weights = np.zeros((num_features, num_classes))
    biases = np.zeros((1, num_classes))

    for epoch in range(epochs):
        # descente de gradient
        z = np.dot(features_matrix, weights) + biases
        # transforme les logits en probablites par classe
        y_pred = softmax(z)

        # dz = erreur. c'est l'ecart entre la prediction et la verite
        dz = y_pred - labels_onehot
        # calcul des gradients par rapport au biais et au poids
        dW = np.dot(features_matrix.T, dz) / num_samples
        db = np.sum(dz, axis=0, keepdims=True) / num_samples

        # mise a jour des poids et des biais avec le learning rate
        weights -= lr * dW
        biases -= lr * db

    return weights, biases


def save_training(weights, biases):
    with open("./model.txt", "w") as f:
        f.write(str(weights) + "|" + str(biases))


def main():
    if len(sys.argv) != 2:
        print("must contain .csv as parameter")
        sys.exit(1)
    print("train")
    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]
    # "./datasets/dataset_train.csv"
    features_matrix, labels_onehot = load_data(sys.argv[1], features, "Hogwarts House")
    weights, biases = train_logistic_regression(features_matrix, labels_onehot, lr=0.01, epochs=1000)
    save_training(weights, biases)


if __name__ == "__main__":
    main()