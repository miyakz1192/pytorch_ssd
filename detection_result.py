import pickle


class DetectionResult:
    def __init__(self, label , score, rect):
        self.label = label
        self.score = score
        self.rect  = rect

    def print(self):
        #http://www.mwsoft.jp/programming/numpy/rectangle.html
        #rect = ((55.251713, 151.57387), 71.9019775390625, 68.363525390625)
        #            x     ,   y       ,  width          ,   height
        print("DETECT: %s(%.2f), %s" % (self.label, self.score, str(self.rect)))


class DetectionResultContainer:
    def __init__(self):
        self.res = []
        pass

    def add(self, label , score, rect):
        self.res.append(DetectionResult(label, score, rect))

    def sort_by_score(self):
        self.res.sort(key=lambda x: x.score)

    def print(self):
        for i in self.res:
            i.print()

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.res, f)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self.res = pickle.load(f)

    def merge(seif, other_container):
        self.res.extend(other_container.res) 

