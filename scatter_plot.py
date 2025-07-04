import matplotlib.pyplot as plt
import pandas as pd

def main():

    df = pd.read_csv("./datasets/dataset_train.csv")

    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    df_features = df[features]
    df_features = df_features.dropna()
    corr_matrix = df_features.corr()

    max_corr = 0
    best_pair = ("", "")
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > max_corr:
                max_corr = corr_value
                best_pair = (features[i], features[j])

    print(best_pair[0], best_pair[1], max_corr)

    plt.figure(figsize=(8, 6))
    plt.scatter(df_features[best_pair[0]], df_features[best_pair[1]], alpha=0.6, color="purple")
    plt.xlabel(best_pair[0])
    plt.ylabel(best_pair[1])
    plt.title(f"Scatter plot : {best_pair[0]} vs {best_pair[1]}")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()