import pandas as pd
import numpy as np
import sys
import os 
 
houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def load_and_prepare_test_data(path, features):
    try:
        df = pd.read_csv(path)
        df = df[features]

        mean = df.mean(skipna=True)
        std = df.std(skipna=True)

        df_normalized = (df - mean) / std

        df_normalized = df_normalized.fillna(0)

        X = df_normalized.to_numpy()
        return X, df.index.to_numpy()
    except Exception as e:
        print(f"Error loading or processing test data: {e}")
        sys.exit(1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta_all):
    m = X.shape[0]
    X_bias = np.c_[np.ones((m, 1)), X]  # add column of ones for the bias
    print(X_bias @ theta_all.T)
    probs = sigmoid(X_bias @ theta_all.T)  # (m, k)
    preds = np.argmax(probs, axis=1)       # (m,)
    return [houses[i] for i in preds]

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 predict.py <test_data.csv> <weights file>")
        sys.exit(1)

    features = [
        "Astronomy", "Ancient Runes", "Herbology"
    ]

    test_path = sys.argv[1]
    weights_path = sys.argv[2]

    if not os.path.isfile(weights_path):
        print(f"Error: weights file '{weights_path}' not found. Please run the training first.")
        sys.exit(1)

    X, ids = load_and_prepare_test_data(test_path, features)

    try:
        theta_all = np.loadtxt(weights_path, delimiter="|")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    predictions = predict(X, theta_all)

    output_df = pd.DataFrame({
        "Index": ids,
        "Hogwarts House": predictions
    })

    output_df.to_csv("houses.csv", index=False)
    print("predictions stored in \"houses.csv\"")

if __name__ == "__main__":
    main()