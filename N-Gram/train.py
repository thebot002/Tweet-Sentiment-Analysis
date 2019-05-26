from preprocessing import tokenize, split
from sklearn.model_selection import train_test_split
from CustomLanguageModel import CustomLanguageModel

print('Processing the data...')
data_path = '../Data/'
# split(data_path, 'tweets.csv')
good_tweets = tokenize(data_path + 'good_tweets.csv')
bad_tweets = tokenize(data_path + 'bad_tweets.csv')

[good_train, good_test, bad_train, bad_test] = train_test_split(good_tweets, bad_tweets)

print('Creation of the models...')
good_model = CustomLanguageModel(good_train)
bad_model = CustomLanguageModel(bad_train)

print('Evaluation of the models...')
true = 0
for i in range(len(good_test)):
    if good_model.score(good_test[i]) > bad_model.score(good_test[i]):
        true += 1
    if good_model.score(bad_test[i]) < bad_model.score(bad_test[i]):
        true += 1

print('accuracy: ' + str(true / (len(good_test) * 2)))