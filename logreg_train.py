import pandas as pd
import numpy as np
import sys

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def load_data(path, features, label_col):
    try:
        df = pd.read_csv(path)
        df = df[features + [label_col]].dropna()

        # Normalization
        X = df[features].to_numpy()
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        y_raw = df[label_col].to_numpy()
        y_onehot = np.zeros((len(y_raw), len(houses)))
        for i, h in enumerate(houses):
            y_onehot[:, i] = (y_raw == h).astype(int)

        return X, y_onehot
    except FileNotFoundError:
        print(f"File {path} not found")
        sys.exit(1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_one_vs_all(X, Y, lr=0.1, epochs=1000):
    m, n = X.shape
    k = Y.shape[1]
    # add column of ones for the bias
    X_bias = np.c_[np.ones((m, 1)), X]  # (m, n+1)
    theta_all = np.zeros((k, n + 1))    # (k, n+1)

    for i in range(k):
        theta = np.zeros(n + 1)  # (n+1,)
        y = Y[:, i]              # (m,)

        for j in range(epochs):
            z = X_bias @ theta       # (m,)
            h = sigmoid(z)           # (m,)
            gradient = X_bias.T @ (h - y) / m  # (n+1,)
            theta -= lr * gradient

        theta_all[i] = theta

    return theta_all  # (k, n+1) 

def save_model(theta_all, path="weights.txt"):
    np.savetxt(path, theta_all, delimiter="|")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <dataset.csv>")
        sys.exit(1)
    features = [
        "Astronomy", "Ancient Runes", "Herbology"
    ]
    dataset_path = sys.argv[1]
    X, Y = load_data(dataset_path, features, "Hogwarts House")
    theta_all = train_one_vs_all(X, Y, lr=0.1, epochs=1000)
    save_model(theta_all)

if __name__ == "__main__":
    main()