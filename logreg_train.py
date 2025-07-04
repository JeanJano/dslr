import pandas as pd
import numpy as np

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def load_data(path, features, label_col):
    df = pd.read_csv(path)
    df = df[features +[label_col]].dropna()

    X = df[features].to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    y_raw = df[label_col].to_numpy()
    y = np.zeros((len(y_raw), len(houses)))
    for i, h in enumerate(houses):
        y[:, i] = (y_raw == h).astype(int)

    return X, y


def main():
    print("train")
    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    X, y = load_data("./datasets/dataset_train.csv", features, "Hogwarts House")

    print(X, "\ny = ", y)

if __name__ == "__main__":
    main()