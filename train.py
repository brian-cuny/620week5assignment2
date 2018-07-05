from __future__ import division
import nltk, re, pprint
import urllib.request
from bs4 import BeautifulSoup

with urllib.request.urlopen('http://news.bbc.co.uk/2/hi/health/2284783.stm') as response:
    html = response.read()
    raw = BeautifulSoup(html, 'html.parser')
    tokens = nltk.word_tokenize(raw.get_text())
    print(tokens[:20])