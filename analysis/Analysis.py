from analysis.Category import Category
import sys
import math

class Analysis:
    lines = []

    def __init__(self, file_path):
        self.file_path = file_path
        self._read_csv()


    def describe(self):
        self._print_format("")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(category.name)
        
        self._print_format("\nCount")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._count(category.value))

        self._print_format("\nMean")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._mean(category.value))

        self._print_format("\nStd")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._std(category.value))

        self._print_format("\nMin")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._min(category.value))

        self._print_format("\n25%")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._quartiles(category.value, 0.25))

        self._print_format("\n50%")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._quartiles(category.value, 0.5))

        self._print_format("\n75%")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._quartiles(category.value, 0.75))

        self._print_format("\nMax")
        for category in Category:
            if Category.ARITHMANCY.value <= category.value <= Category.FLYING.value:
                self._print_format(self._max(category.value))


    def _read_csv(self):
        try:
            with open(self.file_path, 'r') as f:
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
        except FileNotFoundError:
            print("file not found:", self.file_path)
            sys.exit(1)


    def _count(self, feature):
        count = 0
        for line in self.lines:
            feat_value = line[feature]
            if feat_value != "":
                count += 1
        return count


    def _mean(self, feature):
        val = self.get_col_val_float(feature)
        
        mean = sum(val) / len(val)
        return mean


    def _std(self, feature):
        val = self.get_col_val_float(feature)

        mean = self._mean(feature)
        total = 0
        for x in val:
            total += (x - mean) ** 2
        
        variance = total / len(val)
        std = math.sqrt(variance)
        return std


    def _min(self, feature):
        val = self.get_col_val_float(feature)

        minimum = 2147483647
        for x in val:
            if minimum > x:
                minimum = x

        return minimum
    

    def _max(self, feature):
        val = self.get_col_val_float(feature)

        maximum = -2147483648
        for x in val:
            if maximum < x:
                maximum = x

        return maximum


    def _quartiles(self, feature, p):
        val = sorted(self.get_col_val_float(feature))
        n = len(val)

        def get_percentile(p):
            index = p * (n - 1)
            base = int(index)
            reste = index - base

            if base + 1 < n:
                return val[base] + reste * (val[base + 1] - val[base])
            else:
                return val[base]
        
        q = get_percentile(p)

        return q


    def get_col_val_float(self, feature):
        val = []
        for line in self.lines:
            feat_value = line[feature]
            if feat_value != "":
                val.append(float(feat_value))
        return val


    def _format_10(self, val):
        s = str(val)
        if len(s) > 10:
            return s[:9] + "."
        else:
            return f"{s:<10}"


    def _print_format(self, s):
        print(self._format_10(s), end=" ")
