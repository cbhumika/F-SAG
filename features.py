# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:20:38 2019

@author: acer
"""
import pandas as pd
from nltk.corpus import wordnet
from itertools import product
import numpy
from nltk import bigrams
from nltk.corpus import wordnet_ic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
brown_ic = wordnet_ic.ic('ic-brown.dat')

def tf_idf(sen1,sen2):
    documents = [sen1, sen2]
    vectorizer = TfidfVectorizer(use_idf=True)
    sp_m = vectorizer.fit_transform(documents)
    #print(sp_m)
    doc_term_matrix = sp_m.todense()
    #print("dense matrix",doc_term_matrix)
    vectorframe = pd.DataFrame(doc_term_matrix, 
                  columns=vectorizer.get_feature_names(), 
                  index=['ans1', 'ans2'])
    #print(vectorframe)
    return vectorframe

def count_vector(sen1,sen2):
    documents = [sen1, sen2]
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
    #print("sparse matrix",sparse_matrix)
# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    #print("dense matrix",doc_term_matrix)
    vectorframe = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['ans1', 'ans2'])
    #print(vectorframe)
    return vectorframe

def cosine_sim(vector1,vector2):
    arr=cosine_similarity(vector1, vector2)
    val=arr[0][1]
    return val

def jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def manhattan_dis(v1):
     arr = manhattan_distances(v1, sum_over_features=True)
     #print(arr)
     r1=arr[0][1]
     return r1

def euclidean_dis(v1):
    res=euclidean_distances(v1,squared=True)
    #print(res)
    r1=res[0][1]
    return r1

def bigram(sen1,sen2):
    string_bigrams1 =list( bigrams(sen1.split()))
    string_bigrams2 = list(bigrams(sen2.split()))
    #print(*map(' '.join, string_bigrams1), sep=', ')
    #print("one",string_bigrams1)
    #print("two",string_bigrams2)
    count=0
    c=0
    for i in string_bigrams1:
        c=c+1
        for j in string_bigrams2:
            if(i==j):
                count=count+1
    d=count/c
    return d

def wordnet_wup_sim(list1,list2):
  sims = []
  final = []
  #print(list1)
  #print(list2)
  

  for word1 in list1:
    simi =[]
    for word2 in list2:
        sims = []
        #print(word1)
        #print(word2)
        syns1 = wordnet.synsets(word1)
        #print("synset1",syns1)
        #print(list1[0])
        syns2 = wordnet.synsets(word2)
        #print(wordFromList2[0])
        #print("synset2",syns2)
        for sense1, sense2 in product(syns1, syns2):
            #print("sense1",sense1)
            #print("sense2",sense2)
            d = wordnet.wup_similarity(sense1, sense2)
            if d != None:
                sims.append(d)
    
        #print(sims)
        #print(max(sims))
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)


##---------------Final Output---------------##

  similarity_index = numpy.mean(final)
  similarity_index = round(similarity_index , 2)
#print("Sentence 1: ",str1)
#print("Sentence 2: ",str2)
  #print("Similarity index value : ", similarity_index)
    
  return similarity_index

def wordnet_path_sim(list1,list2):
  sims = []
  final = []
  #print(list1)
  #print(list2)
  

  for word1 in list1:
    simi =[]
    for word2 in list2:
        sims = []
        #print(word1)
        #print(word2)
        syns1 = wordnet.synsets(word1)
        #print("synset1",syns1)
        #print(list1[0])
        syns2 = wordnet.synsets(word2)
        #print(wordFromList2[0])
        #print("synset2",syns2)
        for sense1, sense2 in product(syns1, syns2):
            #print("sense1",sense1)
            #print("sense2",sense2)
            d = wordnet.path_similarity(sense1, sense2)
            if d != None:
                sims.append(d)
    
        #print(sims)
        #print(max(sims))
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)


##---------------Final Output---------------##

  similarity_index = numpy.mean(final)
  similarity_index = round(similarity_index , 2)
#print("Sentence 1: ",str1)
#print("Sentence 2: ",str2)
  #print("Similarity index value : ", similarity_index)
    
  return similarity_index

def wordnet_lch_sim(list1,list2):
  sims = []
  final = []
  #print(list1)
  #print(list2)
  

  for word1 in list1:
    simi =[]
    for word2 in list2:
        sims = []
        #print(word1)
        #print(word2)
        syns1 = wordnet.synsets(word1)
        #print("synset1",syns1)
        #print(list1[0])
        syns2 = wordnet.synsets(word2)
        #print(wordFromList2[0])
        #print("synset2",syns2)
        d=None
        for sense1, sense2 in product(syns1, syns2):
            #print("sense1",sense1)
            #print("sense2",sense2)
            pos1=sense1.pos()
            pos2=sense2.pos()
            if(pos1=='s' or pos2=='s'):
                continue
            if(pos1 == pos2):
               d = wordnet.lch_similarity(sense1, sense2,brown_ic)
            #d = wordnet.lch_similarity(sense1, sense2)
            if d != None:
                sims.append(d)
    
        #print(sims)
        #print(max(sims))
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)


##---------------Final Output---------------##

  similarity_index = numpy.mean(final)
  similarity_index = round(similarity_index , 2)
#print("Sentence 1: ",str1)
#print("Sentence 2: ",str2)
  #print("Similarity index value : ", similarity_index)
    
  return similarity_index

def wordnet_res_sim(list1,list2):
  sims = []
  final = []
  #print(list1)
  #print(list2)
  

  for word1 in list1:
    simi =[]
    for word2 in list2:
        sims = []
        #print(word1)
        #print(word2)
        syns1 = wordnet.synsets(word1)
        #print("synset1",syns1)
        #print(list1[0])
        syns2 = wordnet.synsets(word2)
        #print(wordFromList2[0])
        #print("synset2",syns2)
        d=None
        for sense1, sense2 in product(syns1, syns2):
            #print("sense1",sense1)
            #print("sense2",sense2)
            pos1=sense1.pos()
            pos2=sense2.pos()
            if(pos1=='s' or pos2=='s' or pos1=='r' or pos2=='r' or pos1=='a' or pos2=='a'):
                continue
            if(pos1 == pos2):
               d = wordnet.res_similarity(sense1, sense2,brown_ic)
            #d = wordnet.res_similarity(sense1, sense2,brown_ic)
            if d != None:
                sims.append(d)
    
        #print(sims)
        #print(max(sims))
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)


##---------------Final Output---------------##

  similarity_index = numpy.mean(final)
  similarity_index = round(similarity_index , 2)
#print("Sentence 1: ",str1)
#print("Sentence 2: ",str2)
  #print("Similarity index value : ", similarity_index)
    
  return similarity_index

def wordnet_jcn_sim(list1,list2):
  sims = []
  final = []
  #print(list1)
  #print(list2)
  

  for word1 in list1:
    simi =[]
    for word2 in list2:
        sims = []
        #print(word1)
        #print(word2)
        syns1 = wordnet.synsets(word1)
        #print("synset1",syns1)
        #print(list1[0])
        syns2 = wordnet.synsets(word2)
        #print(wordFromList2[0])
        #print("synset2",syns2)
        d=None
        for sense1, sense2 in product(syns1, syns2):
            #print("sense1",sense1)
            #print("sense2",sense2)
            pos1=sense1.pos()
            pos2=sense2.pos()
            if(pos1=='s' or pos2=='s' or pos1=='r' or pos2=='r' or pos1=='a' or pos2=='a'):
                continue
            if(pos1 == pos2):
               d = wordnet.jcn_similarity(sense1, sense2,brown_ic)
            #d = wordnet.res_similarity(sense1, sense2,brown_ic)
            if d != None:
                sims.append(d)
    
        #print(sims)
        #print(max(sims))
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)


##---------------Final Output---------------##

  similarity_index = numpy.mean(final)
  similarity_index = round(similarity_index , 2)
#print("Sentence 1: ",str1)
#print("Sentence 2: ",str2)
  #print("Similarity index value : ", similarity_index)
    
  return similarity_index

def wordnet_lin_sim(list1,list2):
  sims = []
  final = []
  #print(list1)
  #print(list2)
  

  for word1 in list1:
    simi =[]
    for word2 in list2:
        sims = []
        #print(word1)
        #print(word2)
        syns1 = wordnet.synsets(word1)
        #print("synset1",syns1)
        #print(list1[0])
        syns2 = wordnet.synsets(word2)
        #print(wordFromList2[0])
        #print("synset2",syns2)
        d=None
        for sense1, sense2 in product(syns1, syns2):
            #print("sense1",sense1)
            #print("sense2",sense2)
            pos1=sense1.pos()
            pos2=sense2.pos()
            if(pos1=='s' or pos2=='s' or pos1=='r' or pos2=='r' or pos1=='a' or pos2=='a'):
                continue
            if(pos1 == pos2):
               d = wordnet.lin_similarity(sense1, sense2,brown_ic)
            #d = wordnet.lin_similarity(sense1, sense2,brown_ic)
            if d != None:
                sims.append(d)
    
        #print(sims)
        #print(max(sims))
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)


##---------------Final Output---------------##

  similarity_index = numpy.mean(final)
  similarity_index = round(similarity_index , 2)
#print("Sentence 1: ",str1)
#print("Sentence 2: ",str2)
  #print("Similarity index value : ", similarity_index)
    
  return similarity_index