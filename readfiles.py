# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:38:08 2019

@author: acer
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:19:52 2019

@author: acer
"""



import os
import glob
import pandas as pd


def sortKeyFunc(s):
    file_text_name = os.path.splitext(os.path.basename(s))  #you get the file's text name without extension
    return int(file_text_name[0])

def sortKeyFunc1(s):
    file_text_name = os.path.splitext(os.path.basename(s))  #you get the file's text name without extension  
    file_last_num = os.path.basename(file_text_name[0]).split('.')  #you get three elements, the last one is the number. You want to sort it by this number
    return int(file_last_num[1])

def sortKeyFunc2(s):
    file_text_name = os.path.splitext(os.path.basename(s))  #you get the file's text name without extension  
    #print(file_text_name[1])
    file_last_num = os.path.basename(file_text_name[1]).split('.')  #you get three elements, the last one is the number. You want to sort it by this number
    #print(file_last_num[1])
    return int(file_last_num[1])

def score_file():
    file_path="./scores"
    read_files = glob.glob(os.path.join(file_path,"*"))
    read_files.sort(key=sortKeyFunc)
    #print(read_files)
    test=pd.DataFrame()
    for i in read_files:
        read_text = glob.glob(os.path.join(i,"*"))
        read_text.sort(key=sortKeyFunc2)
        #print(read_text)
        for files in read_text:
             score=pd.read_csv(files+'/ave',header=None,delimiter='\t',encoding='UTF-8')
             test=test.append(score)
            
    return test
   

def ref_file():
    #pd.set_option('display.width', 1500)
    ref=pd.read_csv('./answers',header=None,delimiter='\t',encoding='UTF-8')   
    return ref

def ques_file():
    #pd.set_option('display.width', 1500)
    ques=pd.read_csv('./questions',header=None,delimiter='\t',encoding='UTF-8')   
    return ques

def stu_file():
    file_path="./stuans"
    read_files = glob.glob(os.path.join(file_path,"*"))
    read_files.sort(key=sortKeyFunc)
    #print(read_files)
    #s_list=[]
    test=pd.DataFrame()
    for i in read_files:
        read_text = glob.glob(os.path.join(i,"*.txt"))
        read_text.sort(key=sortKeyFunc1)
    #print(read_text)
        for files in read_text:
             stu=pd.read_csv(files,header=None,delimiter='\t',encoding='UTF-8')
             #s_list.append(stu)
             test=test.append(stu)
   # print(test)        
    return test
#print(len(read_files))
#print(len(read_text))
#print(read_text)
   
def load_embeddings(file_name):

    embeddings = {}

    input_file = open(file_name, 'r',encoding='utf8')
    for line in input_file:
        tokens = line.split('\t')
        tokens[-1] = tokens[-1].strip()
        for i in range(1, len(tokens)):
            tokens[i] = float(tokens[i])
        embeddings[tokens[0]] = tokens[1:-1]

    return embeddings




         
       