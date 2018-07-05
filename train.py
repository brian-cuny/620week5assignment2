import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import random
import pickle

def feature(topic, i):
    with open(f'/home/brian/Desktop/DataScience/620/620week5assignment2/Train/{topic}/{i}', 'rb') as response:
        html = response.read()
        raw = BeautifulSoup(html, 'html.parser')
        tokens = nltk.word_tokenize(raw.get_text())
        tokens = set(w.lower() for w in tokens if w.isalpha()) #only words
        tokens = tokens.difference(stopwords.words('english')) #keep import words

        features = {}
        for t in tokens:
            features[t] = True
        return features

def initial_read():
    featuresets_bands = [(feature('Bands', i), 'Bands') for i in range(1,61)]
    featuresets_bio = [(feature('BioMedical', i), 'BioMedical') for i in range(1,61)]
    featuresets_goats = [(feature('Goats', i), 'Goats') for i in range(1,58)]
    featuresets_sheeps = [(feature('Sheep', i), 'Sheep') for i in range(1,57)]

    full_set = featuresets_bands + featuresets_bio + featuresets_goats + featuresets_sheeps
    random.shuffle(full_set)

    file_name = 'data'
    file_object = open(file_name, 'wb')
    pickle.dump(full_set, file_object)
    file_object.close()

# initial_read()
file_name = 'data'
file_object = open(file_name, 'rb')
full_set = pickle.load(file_object)

random.shuffle(full_set)
size = int(len(full_set) * 0.1)

train_set, test_set = full_set[size:], full_set[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

tree_classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(tree_classifier, test_set))
