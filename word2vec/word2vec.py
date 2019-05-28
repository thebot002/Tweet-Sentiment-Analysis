import numpy as num
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from tqdm import tqdm

class w2v:
    def __init__(self, x_data, y_data):
        self.w2v_model = Word2Vec(x_data, size=100, window=5, workers=8)
        print('Training of word2vec')
        self.w2v_model.train(x_data, total_examples=len(x_data), epochs=2)

        self.reg = LinearSVC()
        x_vect = []
        t = tqdm(total=len(x_data))
        for sen in x_data:
            t.update()
            x_vect.append(self.sen2vec(sen))
        t.close()
        self.reg.fit(x_vect, y_data)


    def sen2vec(self, sen):
        M = []
        for w in sen:
            try:
                M.append(self.w2v_model.wv[w])
            except:
                continue
        M = num.array(M)
        v = M.sum(axis=0)
        vec = v / num.sqrt(((v ** 2).sum()))
        return vec.tolist()

    def evaluate(self, x_data, y_data):
        true = 0
        t = tqdm(total=len(x_data))
        for i in range(len(x_data)):
            t.update()
            pred = self.reg.predict(self.sen2vec(x_data[i]))
            if pred == y_data[i]:
                true += 1
        t.close()
        print('\nAccuracy: ' + str(true/len(x_data)))

