from analysis.Analysis import Analysis
from analysis.Category import Category
import matplotlib.pyplot as plt

def show_histogram(plot, name, x, y):
    bars = plot.bar(x, y, color='skyblue')
    for bar, val in zip(bars, y):
        offset = 0.1 if val >= 0 else -0.4
        va = 'bottom' if val >= 0 else 'top'
        plot.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.3f}", ha='center', va=va)

    plot.axhline(0, color='black', linewidth=0.8)
    plot.set_title(name)
    plot.set_xticks(range(len(x)))
    plot.set_xticklabels(x, rotation=45, ha='right', fontsize=8)
    plot.set_ylim(-600, 2000)


def main():
    analysis = Analysis("./datasets/dataset_train.csv")
    gryffindor = analysis.get_std_by_house("Gryffindor")
    hufflepuff = analysis.get_std_by_house("Hufflepuff")
    slytherin = analysis.get_std_by_house("Slytherin")
    ravenclaw = analysis.get_std_by_house("Ravenclaw")
    matieres = ["Arithmancy", "Astronomy", "Herbology", "Defense the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "flying"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    show_histogram(axes[0, 0], "Ravenclaw", matieres, ravenclaw)
    show_histogram(axes[0, 1], "Gryffindor", matieres, gryffindor)
    show_histogram(axes[1, 0], "Hufflepuff", matieres, hufflepuff)
    show_histogram(axes[1, 1], "Slytherin", matieres, slytherin)

    plt.show()


if __name__ == "__main__":
    main()

# les cours avec une distributions homogenes sont :
# artihmancy
# transfiguration
# potions
# Care of Magical Creatures
# charms
