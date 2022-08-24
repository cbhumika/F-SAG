# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:32:13 2019

@author: acer
"""
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def remove_punctuations(s):
    # define punctuation
    punctuations = '''-,.;?'"'''
    for punct in punctuations:
        s = s.replace(punct, '')
    #print(s)
    return s

def tokenization(s):
    test = word_tokenize(s)
    return test

def lower_split(ref):
    #print(type(ref))
    
   # test=ref[0].lower().split()
   # print(test)
    test=ref.apply(lambda x: x.lower().split())
   # print(test)
    return test

def remove_stopwords(ref):
#print(t)
   stop = stopwords.words('english')
   #print(stop)
   test = ref.apply(lambda x:' '.join([item for item in x if item not in stop]))
#print(test)
   return test

def stemming(ref):
   # print(ref)
    ps = PorterStemmer()
    test = ref.apply(lambda x:' '.join([ps.stem(item) for item in x]))
    return test

def lemmatize(ref):    
    #print(ref)
    lmtz = WordNetLemmatizer()
    test = ref.apply(lambda x:' '.join([lmtz.lemmatize(item,'v') for item in x]))
    return test