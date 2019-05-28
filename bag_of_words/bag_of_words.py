import collections
import math
from tqdm import tqdm


class bow:
    def __init__(self, data):
        self.bow = collections.defaultdict(lambda: 0)

        t = tqdm(total=len(data))
        for sentence in data:
            t.update()
            for word in sentence:
                self.bow[word] += 1
        t.close()

    def score(self, sentence, voc):
        score = 0
        for word in sentence:
            prob = (self.bow[word] + 1)/(len(self.bow) + voc)
            score += math.log(prob)
        return score

    def __len__(self):
        return len(self.bow)
