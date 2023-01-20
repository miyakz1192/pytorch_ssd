import pickle

class DetectionResultRectangle:
    def __init__(self, rect_taple):
        first_layer= list(rect_taple)
        pos = list(first_layer[0])

        self.x = pos[0]
        self.y = pos[1]

        self.width  = first_layer[1]
        self.height = first_layer[2]

    def to_s(self):
        x = int(self.x)
        y = int(self.y)
        w = int(self.width)
        h = int(self.height)
        return f"x={x},y={y},w={w},h={h}"


class DetectionResult:
    def __init__(self, label , score, rect_taple):
        self.label = label
        self.score = score
        self.rect = DetectionResultRectangle(rect_taple)

    def print(self):
        #http://www.mwsoft.jp/programming/numpy/rectangle.html
        #rect = ((55.251713, 151.57387), 71.9019775390625, 68.363525390625)
        #            x     ,   y       ,  width          ,   height
        print("DETECT: %s(%.2f), %s" % (self.label, self.score, self.rect.to_s()))


class DetectionResultContainer:
    def __init__(self):
        self.res = []
        pass

    def add(self, label , score, rect):
        self.res.append(DetectionResult(label, score, rect))

    def sort_by_score(self):
        self.res.sort(key=lambda x: -x.score)

    def print(self):
        for i in self.res:
            i.print()

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.res, f)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self.res = pickle.load(f)

    def merge(self, other_container):
        self.res.extend(other_container.res) 
        
