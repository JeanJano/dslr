from analysis.Analysis import Analysis

def main():

    analysis = Analysis("./datasets/dataset_train.csv")

    lines = []
    with open("./datasets/dataset_train.csv", 'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            if i == 0:
                i += 1
                continue

            line_split = line.split(",")
            print(line_split)
            i += 1
            if i > 1:
                break

    # print(lines[0])
        # for i in data:
        #     print(i)
        #     if i == "\n":
        #         break

if __name__ == "__main__":
    main()