import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    df = pd.read_csv("./datasets/dataset_train.csv")
    df = df.dropna(subset=features + ["Hogwarts House"])
    houses = df["Hogwarts House"].unique()

    house_colors = {
        'Gryffindor': 'red',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'gold',
        'Slytherin': 'green'
    }

    houses = list(house_colors.keys())

    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        stds = []
        for house in houses:
            scores = df[df["Hogwarts House"] == house][feature]
            stds.append(np.std(scores))

        bars = ax.bar(houses, stds, color=[house_colors[h] for h in houses])
        ax.set_title(feature, fontsize=10)
        ax.set_ylabel("Écart-type")
        ax.set_ylim(0, max(stds) * 1.2)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}",
                    ha='center', va='bottom', fontsize=8)

    for i in range(len(features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# le cours avec la distribution la plus homogene est:
# care of magical creature
