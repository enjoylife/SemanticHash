import os
import sys
import string
import unicodedata
from collections import Counter
from porter import PorterStemmer
from itertools import takewhile

TABLE = string.maketrans("","")
STEMMER = PorterStemmer()

def gen_stops():
    english_ignore = []
    with open('stoplist.txt',  'r') as stops:
        for word in stops:
            english_ignore.append(word.strip())
    return frozenset(english_ignore) # faster??

STOPLIST =  gen_stops()

def ngrams(tokens, MIN_N, MAX_N):
    """ Params: iterable of tokens, the smallest and the largest nrgram you want
    If both Min_N and MAX_N are 1, it just yields the same iterable."""
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+MIN_N, min(n_tokens, i+MAX_N)+1):
            yield " ".join(tokens[i:j]) # python string concat slow

def text_processer(doc,punc=True):
    """ Alot of python magic and helpers in this list comprehension
     If this is one area where a more precise C implementation would be
     alot faster but more work."""
    # get ride of weird unicode that could pop up 
    if isinstance(doc, unicode):
        doc = unicodedata.normalize('NFKD',doc).encode('ascii','ignore')
    if  not punc:
        # don't want puncuation, vaporize it!! 
        doc = doc.translate(TABLE, string.punctuation)
    return [STEMMER.stem(x,0, len(x)-1) for x in ngrams((doc.lower().split()),1,2) if x not in STOPLIST]

def eat_heading(f):
    check = False
    while not check:
        line = f.readline()
        if line[0:5] == "Lines":
            check = True
            print line[7:]

def msgreader(file):
    while True:
        header = list(takewhile(lambda x: x.strip(), file))
        if not header: break
        header_dict = {k: v.strip() for k,v in (line.split(":", 1) for line in header)}
        line_count = int(header_dict['Lines'])
        message = [next(file) for i in xrange(line_count)] # or islice..
        yield message

c = Counter()
a = []

for root, dirs, files, in os.walk('../20news-bydate-test'):
    for name in files:
        with open(root +'/'+ name,'r') as f:
            print root + '/' +name
    for name in dirs:
        pass
        #print name


