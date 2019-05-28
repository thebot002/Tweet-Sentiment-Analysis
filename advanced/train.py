from preprocessing import tokenize, read_preprocessed
import bag_of_words
import numpy as np
from sklearn.model_selection import train_test_split
import my_lstm

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

x_vector = good_tweets + bad_tweets
y_vector = (np.zeros(len(good_tweets)).tolist()) + (np.ones(len(bad_tweets)).tolist())

[x_train, x_test, y_train, y_test] = train_test_split(x_vector, y_vector, shuffle=True)

print('Creating vocab...')
bow = bag_of_words.bow(x_train)

print('Retrieving max input size...')
max_input_size = 1
for sentence in x_vector:
    if len(sentence) > max_input_size:
        max_input_size = len(sentence)

print('Creation of the model...')
model = my_lstm.MyLSTM(len(bow), [x_train, y_train], max_input_size)

print('Evaluation...')
print(model.score([x_test, y_test]))
