import csv
import re
from nltk.corpus import stopwords


# To separate our file of tweets in two different files
def split(path, file_name):
    good_out = open(path + "good_tweets.csv", "w+")
    bad_out = open(path + "bad_tweets.csv", "w+")

    seen = 1
    with open(path + file_name, 'r') as f:
        reader = csv.reader(f)
        # reader.__next__()

        for line in reader:
            seen += 1
            sentiment = line[0]
            sentence = line[-1]

            if (sentiment == "0"):
                bad_out.write(sentence+"\n")
            else:
                good_out.write(sentence+"\n")

            if (seen % 10000 == 0):
                print(seen)

    good_out.close()
    bad_out.close()


def preprocess(string):
    # EMOJIS
    string = re.sub(r":\)", "emojihappy1", string)
    string = re.sub(r":P", "emojihappy2", string)
    string = re.sub(r":p", "emojihappy3", string)
    string = re.sub(r":>", "emojihappy4", string)
    string = re.sub(r":3", "emojihappy5", string)
    string = re.sub(r":D", "emojihappy6", string)
    string = re.sub(r" XD ", "emojihappy7", string)
    string = re.sub(r" <3 ", "emojihappy8", string)

    string = re.sub(r":\(", "emojisad9", string)
    string = re.sub(r":<", "emojisad10", string)
    string = re.sub(r":<", "emojisad11", string)
    string = re.sub(r">:\(", "emojisad12", string)

    # MENTIONS "(@)\w+"
    string = re.sub(r"(@)\w+", "mentiontoken", string)

    # WEBSITES
    string = re.sub(r"http(s)*:(\S)*", "linktoken", string)

    # STRANGE UNICODE \x...
    string = re.sub(r"\\x(\S)*", "", string)

    # General Cleanup and Symbols
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    stop_words = set(stopwords.words('english'))
    for w in stop_words:
        string = re.sub((" "+w+" "), " ", string)

    return string.strip().lower()


def tokenize(filename):
    dataset = []
    with open(filename) as file:
        for line in file:
            dataset.append(preprocess(line).split())
    return dataset
