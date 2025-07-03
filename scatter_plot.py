from analysis.Analysis import Analysis
from analysis.Category import Category
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print("scatter plot")

    # analysis = Analysis("./datasets/dataset_train.csv")
    # arithmancy = analysis.get_col_val_float(Category.ARITHMANCY.value)
    # astronomy = analysis.get_col_val_float(Category.ASTRONOMY.value)
    # print(arithmancy, astronomy)

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


if __name__ == "__main__":
    main()