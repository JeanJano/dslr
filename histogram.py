from analysis.Analysis import Analysis
from analysis.Category import Category

def main():
    analysis = Analysis("./datasets/dataset_train.csv")
    ravenclaw = analysis.get_means_by_house("Ravenclaw")
    gryffindor = analysis.get_means_by_house("Gryffindor")
    hufflepuff = analysis.get_means_by_house("Hufflepuff")
    slytherin = analysis.get_means_by_house("Slytherin")
    print(ravenclaw,"\n", gryffindor,"\n", hufflepuff,"\n", slytherin)


if __name__ == "__main__":
    main()
