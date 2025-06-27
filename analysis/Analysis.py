from analysis.Category import Category

class Analysis:
    lines = []

    def __init__(self, file_path):
        self.file_path = file_path
        self._read_csv()

        # print(self.lines)

    def show(self):
        # count = self._count(Category.CARE_OF_MAGICAL_CREATURES.value)
        # print(count)
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                print(category.name)

    def _read_csv(self):
        with open("./datasets/dataset_train.csv", 'r') as f:
            line = f.readlines()
            i = 0
            for l in line:
                if i == 0:
                    i += 1
                    continue
                
                line_split = l.split(",")
                line_split[Category.FLYING.value] = line_split[Category.FLYING.value][:-1]
                self.lines.append(line_split)
                i += 1
                # if i > 1:
                #     break


    def _count(self, feature):
        count = 0
        for line in self.lines:
            feat_value = line[feature]
            if (feat_value != ""):
                count += 1
        return count

