import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    features = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    df = pd.read_csv("./datasets/dataset_train.csv")
    df = df[["Hogwarts House"] + features].dropna()
    sns.pairplot(df, corner=True, hue="Hogwarts House")
    plt.show()


if __name__ == "__main__":
    main()


# Permet de savoir si on peut distinguer visuellement les features grace aux maisons
# Quand une feature (ou combinaison de features) permet de bien distinguer les maisons, cela veut dire :
#   -Il existe des régions claires dans l’espace des données (features) où chaque maison est concentrée.
#   -Donc, la régression logistique pourra tracer une frontière entre ces régions et prédire à quelle maison appartient un nouvel élève en fonction de ses scores.

# Pour la logistic regression on va utiliser les features:
#   - ancient runes
#   - astronomy
#   - herbology