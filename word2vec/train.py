from preprocessing import tokenize, read_preprocessed
import numpy as np
from word2vec import w2v
from sklearn.model_selection import train_test_split

use_preprocessed = True

if not use_preprocessed:
    print('Processing the data...')
    data_path = '../Data/'
    # split(data_path, 'tweets.csv')
    good_tweets = tokenize(data_path + 'good_tweets.csv')
    bad_tweets = tokenize(data_path + 'bad_tweets.csv')
else:
    print('Loading preprocessed data...')
    good_tweets = read_preprocessed('good_tweets')
    bad_tweets = read_preprocessed('bad_tweets')

print('Creation of x and y vectors...')
x_vector = good_tweets + bad_tweets
y_vector = (np.zeros(len(good_tweets)).tolist()) + (np.ones(len(bad_tweets)).tolist())

[x_train, x_test, y_train, y_test] = train_test_split(x_vector, y_vector, shuffle=True)

print('Creating the model...')
model = w2v(x_train, y_train)

print('Evaluation of the model...')
model.evaluate(x_test, y_test)
