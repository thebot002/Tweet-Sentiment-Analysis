from preprocessing import tokenize, read_preprocessed
from sklearn.model_selection import train_test_split
from bag_of_words import bow
from tqdm import tqdm

use_preprocessed = False

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

[good_train, good_test, bad_train, bad_test] = train_test_split(good_tweets, bad_tweets)

print('Creating the models...')
good_bow = bow(good_train)
bad_bow = bow(bad_train)

print('Evaluation of the models...')
true = 0
voc = len(good_bow) + len(bad_bow)
t = tqdm(total=len(good_test))
for i in range(len(good_test)):
    t.update()
    if good_bow.score(good_test[i], voc) > bad_bow.score(good_test[i], voc):
        true += 1
    if good_bow.score(bad_test[i], voc) < bad_bow.score(bad_test[i], voc):
        true += 1
t.close()
print('Accuracy: ' + str(true/(len(good_test) * 2)))
