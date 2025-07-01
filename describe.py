from analysis.Analysis import Analysis
import sys

def main():

    if len(sys.argv) != 2:
        print("Must have one argument")
        sys.exit(1)

    # "./datasets/dataset_train.csv"
    analysis = Analysis(sys.argv[1])
    analysis.describe()

if __name__ == "__main__":
    main()