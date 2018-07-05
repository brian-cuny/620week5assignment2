from __future__ import division
import nltk, re, pprint
from nltk.corpus import stopwords
import urllib.request
from bs4 import BeautifulSoup

def feature(i):
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/brian-cuny/620week5assignment2/master/Train/Bands/{i}') as response:
        html = response.read()
        raw = BeautifulSoup(html, 'html.parser')
        tokens = nltk.word_tokenize(raw.get_text())
        tokens = set(w.lower() for w in tokens if w.isalpha()) #only words
        tokens = tokens.difference(stopwords.words('english')) #keep import words

        features = {}
        for t in tokens:
            features[t] = True
        return features

featuresets = [(feature(i), 'Bands') for i in range(1,5)]


print(featuresets)