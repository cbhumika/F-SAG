# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:23:22 2019

@author: acer
"""
from scipy import spatial
import numpy as np
from config import *
from readfiles import *
from preprocessing import *
from preprocessing import tokenization
from config import *
from align import *


    
def r_ques(sen1):
    #print(ques)
    rtokens=tokenization(sen1)
    rtoken=rtokens[0]
    for q in ques:
        tokens3=tokenization(q)
        #print(tokens3)
        f2=tokens3[0]
        #print(f2)
        #print(rtoken)
        if rtoken == f2:
            f2=int(f2)
            #print(f2)
            return tokens3
        

def question_demoting(sent1,sent2):
    sen1=[]
    sen2=[]
    sentence1=tokenization(sent1)
    #print(sentence1)
    sentence2=tokenization(sent2)
    #print(sentence2)
    question=r_ques(sent1)
    #print(question)
    for i in range(len(sentence1)):
        if sentence1[i] not in question:
          sen1.append(sentence1[i])
    for j in range(len(sentence2)):
        if sentence2[j] not in question:
          sen2.append(sentence2[j])
  
    #print("sen1",sen1)
    #print("sen2",sen2)
    return sen1,sen2


def vector_sum(vectors):

    n = len(vectors)
    d = len(vectors[0])

    s = []
    for i in range(d):
        s.append(0)
    s = np.array(s)

    for vector in vectors:
        s = s + np.array(vector)

    return list(s)


def cosine_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


def sts_cvm(sentence1, sentence2,embeddings):
  
    sentence1_lemmas,sentence2_lemmas=question_demoting(sentence1,sentence2)
    sentence1_content_lemma_embeddings = []
    for lemma in sentence1_lemmas:
        if lemma in embeddings:
            sentence1_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])

    
    sentence2_content_lemma_embeddings = []
    for lemma in sentence2_lemmas:
        if lemma.lower() in embeddings:
            sentence2_content_lemma_embeddings.append(+
                                            embeddings[lemma.lower()])

    if sentence1_content_lemma_embeddings == \
                       sentence2_content_lemma_embeddings:
        return 1
    elif sentence1_content_lemma_embeddings == [] or \
         sentence2_content_lemma_embeddings == []:
        return 0
    
    sentence1_embedding = vector_sum(sentence1_content_lemma_embeddings)
    sentence2_embedding = vector_sum(sentence2_content_lemma_embeddings)
    
    return cosine_similarity(sentence1_embedding, sentence2_embedding)

def sts_w_cvm(sentence1_lemmas, sentence2_lemmas,embeddings):
    sentence1_content_lemma_embeddings = []
    for lemma in sentence1_lemmas:
        if lemma in embeddings:
            sentence1_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])

    
    sentence2_content_lemma_embeddings = []
    for lemma in sentence2_lemmas:
        if lemma.lower() in embeddings:
            sentence2_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])

    if sentence1_content_lemma_embeddings == \
                       sentence2_content_lemma_embeddings:
        return 1
    elif sentence1_content_lemma_embeddings == [] or \
         sentence2_content_lemma_embeddings == []:
        return 0
    
    sentence1_embedding = vector_sum(sentence1_content_lemma_embeddings)
    sentence2_embedding = vector_sum(sentence2_content_lemma_embeddings)
    
    return cosine_similarity(sentence1_embedding, sentence2_embedding)

question = "How are infix expressions evaluated by computers?"
ref_answer = "First, they are converted into postfix form, " + \
             "followed by an evaluation of the postfix expression."
student_response = "computers usually convert infix expressions to postfix " +\
                   "expression and evaluate them using a stack."
                   
c=sts_cvm(ref_answer, student_response,embeddings)