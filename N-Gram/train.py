from preprocessing import tokenize, read_preprocessed
from sklearn.model_selection import train_test_split
from CustomLanguageModel import CustomLanguageModel
import tqdm

use_preprocessed = True
if use_preprocessed:
    print('Loading the data...')
    good_tweets = read_preprocessed('good_tweets')
    bad_tweets = read_preprocessed('bad_tweets')
else:
    print('Processing the data...')
    data_path = '../Data/'
    # split(data_path, 'tweets.csv')
    good_tweets = tokenize(data_path + 'good_tweets.csv')
    bad_tweets = tokenize(data_path + 'bad_tweets.csv')

[good_train, good_test, bad_train, bad_test] = train_test_split(good_tweets, bad_tweets)

n = 1
# coef = 1
# print('Creation of the models...')
# good_model = CustomLanguageModel(good_train, n=n, coef=1)
# bad_model = CustomLanguageModel(bad_train, n=n, coef=1)
#
# print('Evaluation of the models...')
# true = 0
# for i in range(len(good_test)):
#     print(str(i) + '/' + str(len(good_test)))
#     if good_model.score(good_test[i]) > bad_model.score(good_test[i]):
#         true += 1
#     if good_model.score(bad_test[i]) < bad_model.score(bad_test[i]):
#         true += 1
#
# print('accuracy ' + str(n) + '-gram, coef=1: ' + str(true / (len(good_test) * 2)))
#
#################################################
# coef = 0.4
print('Creation of the models...')
good_model = CustomLanguageModel(good_train, n=n, coef=0.4)
bad_model = CustomLanguageModel(bad_train, n=n, coef=0.4)

print('Evaluation of the models...')
true = 0
t = tqdm.tqdm(total=len(good_test))
for i in range(len(good_test)):
    t.update()
    if good_model.score(good_test[i]) > bad_model.score(good_test[i]):
        true += 1
    if good_model.score(bad_test[i]) < bad_model.score(bad_test[i]):
        true += 1
t.close()

print('accuracy ' + str(n) + '-gram, coef=0.4: ' + str(true / (len(good_test) * 2)))