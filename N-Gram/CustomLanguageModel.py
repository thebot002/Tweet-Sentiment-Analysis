import math, collections
import tqdm

class CustomLanguageModel:

    def __init__(self, corpus, n=2, coef=0.4):
        """Initialize your data structures in the constructor."""
        self.n = n  # n-gram
        self.counts = []
        for i in range(0, self.n):
            self.counts.append(collections.defaultdict(lambda: 0))
        self.total = 0
        self.train(corpus)
        self.coef = coef

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        t = tqdm.tqdm(total=len(corpus))
        for sentence in corpus:
            t.update()
            for i in range(0, len(sentence)):
                self.total += 1
                tup = []
                for l in range(0, self.n if i >= self.n else i + 1):
                    word = sentence[i - l]
                    tup.insert(0, word)

                    self.counts[l][tuple(tup)] += 1
        t.close()
        pass

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        for i in range(self.n - 1, len(sentence)):
            l = self.n - 1
            while l >= 0:
                if self.counts[l][tuple(sentence[i - l: i+1])] != 0 and l > 0:
                    score += math.log(self.counts[l][tuple(sentence[i - l: i+1])])
                    score -= math.log(self.counts[l-1][tuple(sentence[i - l: i])])
                    break
                elif l == 0:
                    score += math.log(self.counts[l][tuple(sentence[i - l: i+1])] + 1)
                    score -= math.log(self.total + len(self.counts[l]))
                    break
                else:
                    l -= 1
                    score += math.log(self.coef)
        return score
