import sys
import pandas as pd
import numpy as np

def load_data(path, features, label_col):
    df = pd.read_csv(path)
    df = df[features +[label_col]].dropna()

    features_matrix = df[features].to_numpy()
    features_matrix = (features_matrix - features_matrix.mean(axis=0)) / features_matrix.std(axis=0)

    return features_matrix


def parse_matrix_block(text_block):
    lines = text_block.strip().strip('[]').split('\n')
    data = []
    for line in lines:
        line = line.replace('[', '').replace(']', '').strip()
        datas = line.split(" ")
        row = []
        for d in datas:
            if d != "":
                row.append(float(d))
        data.append(row)
    return data


def load_training(path):
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        parts = content.strip().split("|")

        weights = parse_matrix_block(parts[0])
        biases = parse_matrix_block(parts[1])

        return weights, biases
    except FileNotFoundError:
        print("file not found:", path)
        sys.exit(1)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict(features_matrix, weights, biases):
    z = np.dot(features_matrix, weights) + biases
    probs = softmax(z)
    return np.argmax(probs, axis=1)


def save_predict(prediction):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    with open("houses.csv", 'w') as f:
        file = "Index, Hogwarts House\n"
        i = 0
        for house in prediction:
            file += f"{i}, {houses[house]}\n"
            i += 1

        f.write(file)


def main():
    print("predict")

    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    features_matrix = load_data("./datasets/dataset_train.csv", features, "Hogwarts House")
    weights, biases = load_training("./model.txt")
    prediction = predict(features_matrix, weights, biases)
    print(prediction, len(prediction))
    save_predict(prediction)


if __name__ == "__main__":
    main()