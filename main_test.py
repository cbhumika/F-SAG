# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:35:47 2019

@author: acer
"""


import pandas as pd
import numpy as np
from readfiles import *
from preprocessing import remove_punctuations
from preprocessing import lower_split
from preprocessing import remove_stopwords
from preprocessing import lemmatize
from preprocessing import tokenization
from features import *
from metrics import *
from sklearn import preprocessing
from visualize import *
from semantic_feature import *
from sul_freatures import *

# Read reference answer, student answer, and assignment questions
ref=ref_file()
stu=stu_file()
score=score_file()
ques=ques_file()

# Apply preprocessing to reference answers
ref= ref[0].apply(remove_punctuations)
ref=lower_split(ref)
ref=remove_stopwords(ref)
ref=lower_split(ref)
ref=lemmatize(ref)
#np.savetxt(r'refFinal.txt', ref.values, fmt='%s')

# Apply preprocessing to questions answers
ques=ques[0].apply(remove_punctuations)
ques=lower_split(ques)
ques=remove_stopwords(ques)
ques=lower_split(ques)
ques=lemmatize(ques)
#np.savetxt(r'quesFinal.txt', ref.values, fmt='%s')
#print(ques)

# Apply preprocessing to student answers
stu=stu[0].apply(remove_punctuations)
#np.savetxt(r'test2.txt', test.values, fmt='%s')
stu=lower_split(stu)
stu=remove_stopwords(stu)
stu=lower_split(stu)
stu=lemmatize(stu) 
#np.savetxt(r'stuFinal.txt', stu.values, fmt='%s')  
#print(stu)

# load semantic embedding of words
embeddings = {}
if embeddings == {}:
        print ('loading embeddings...')
        embeddings = load_embeddings('Resources/EN-wform.w.5.cbow.neg10.400.subsmpl.txt')
        print ('done')  

print("Embeddings:",embeddings)



c=-1
#counter = 0
dframe=pd.DataFrame()
"""Type_new = pd.Series([]) 
Type_new2 = pd.Series([]) 
Type_new3 = pd.Series([])
Type_new4 = pd.Series([])
Type_new5 = pd.Series([]) 
Type_new6 = pd.Series([])
Type_new7 = pd.Series([])
Type_new8 = pd.Series([])
Type_new12 = pd.Series([])
Type_new9 = pd.Series([])
Type_new10 = pd.Series([])
Type_new11 = pd.Series([])

Type_newlsa=pd.Series([])"""
Type_new_length=pd.Series([])


# Calculate Similarity score for student answer and reference answer
d=0
p=0

for i in ref:
    #print(counter)
    #print(i)
    tokens=tokenization(i)
    #print(tokens)
    f=tokens[0]
    f=int(f)
   
    #print("ref",f)
   # counter=counter+1
    #print("*************Reference Answer**************")
    #print("counter",counter) 
    #Type_new_lsa=pd.Series([])
    cn=0
    for j in stu:
        
        tokens2=tokenization(j)
        f1=tokens2[0]
        f1=int(f1)
        if(f==f1):
            #print("stu",f1)
            c=c+1
            
            #print("...............student Answer-------------------")
            #print(j)
            #print("hello")
            #l=length_ratio(tokens, tokens2)
            sim=sts_cvm(i,j,embeddings)
            #print(l)
            Type_new_length[c]=sim
            #print(Type_new_length)
            cn=cn+1
            """Type_new_lsa[cn]=j
            df=count_vector(i,j)
            #tif=tf_idf(i,j)
            #print(df)
            arr=cosine_sim(df,df)
            
            #print(val)
            Type_new[c]=arr
            #print(Type_new[c])
            scr=jaccard_sim(i, j)
            #print(scr)
            Type_new2[c]=scr
            res=manhattan_dis(df)
            Type_new3[c]=res
            r1=euclidean_dis(df)
            Type_new4[c]=r1
            r2=bigram(i,j)
            #print(r2)
            Type_new5[c]=r2
            
            s1=wordnet_wup_sim(tokens,tokens2)
            Type_new6[c]=s1
            
            s2=wordnet_path_sim(tokens,tokens2)
            Type_new7[c]=s2
            
            s3=wordnet_lch_sim(tokens,tokens2)
            Type_new8[c]=s3
            Type_new12[c]=f1
            print(Type_new8) 
            s4=wordnet_res_sim(tokens,tokens2)
            Type_new9[c]=s4
            print(Type_new9)
            s5=wordnet_jcn_sim(tokens,tokens2)
            Type_new11[c]=s5
            print(Type_new11)
            s6=wordnet_lin_sim(tokens,tokens2)
            Type_new11[c]=s6
            print(Type_new11)"""
        else:
            continue;
    #print(Type_new_lsa)
    """temp=pd.DataFrame()
    temp.insert(0,"ass",Type_new_lsa)
    res=lsa(i,temp,d)
    p=p+1
    for i in range(len(res)):
        if (i == 0):
            continue
        Type_newlsa[d]=res[i]
        d=d+1
    #print ("assingnment no",p)
    #print(Type_lsa)"""
    
    #break;
    #lsa(i,temp[0].astype(str))
        #lsa(i,)

#print(Type_new5)
#list1=['a','b','c']
#data=pd.read_csv("Simfeatures8.csv") 
frame = pd.DataFrame()
"""dframe.insert(0, "Cosine_Sim", Type_new) 
dframe.insert(1, "Jaccard_Sim", Type_new2)
dframe.insert(2, "Manhattan_Dis", Type_new3)
dframe.insert(3, "Euclidean_Dis", Type_new4)
dframe.insert(4, "Bi_gram", Type_new5) 
#dframe[4,'Bi_gram']=Type_new5

dframe.insert(5, "Wordnet_wup_sim", Type_new6)
#print(df)
dframe.insert(6, "Wordnet_path_sim", Type_new7)
dframe.insert(7, "Wordnet_lch_sim", Type_new8)
dframe.insert(1, "QuesAnsNo", Type_new12)
dframe.insert(8, "wordnet_res_sim", Type_new9)
dframe.insert(9, "Wordnet_jcn_sim", Type_new10)"""
#dframe.insert(10, "Word_jcn_sim", Type_new11)

#dframe.insert(0, "lsa2", Type_newlsa)
frame.insert(0, "embed_sim", Type_new_length)
#print(dframe)
"""with open('features.csv', 'a') as f:
    dframe.to_csv(f, header=True)"""
#print(dframe)
dframe.to_csv('Sul.csv')



#read data from python dataframe
#data=pd.read_csv("Simfeatures8.csv") 
#dframe = pd.DataFrame(data)
#print(score[0])
#dframe.insert(11, "score1", score[0])
#score.to_csv('Simfeatures10.csv')
# metric calculation code
#x=dframe['lsa'].fillna(0).values

#x=x.tolist()
#z=x.isna().sum()
#print(z)
#dframe['wordnet_res_sim'].plot(kind='bar')
#xp = dframe[['lsa']].fillna(0).values.astype(float)

# Create a minimum and maximum processor object
#min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
#x_scaled = min_max_scaler.fit_transform(xp)

# Run the normalizer on the dataframe
#df_normalized = pd.DataFrame(x_scaled)
#x=df_normalized.round(7)
# Plot the dataframe
#df_normalized.plot(kind='bar')

"""
z=frame["length"]
print(z)
y=score[0].tolist()
#print(y)
cor,pval = pearson_r(z,y)
#cor,pval = spearman_r(z,y)
print("Person r Correlation and p val",cor,pval)       
scatter_plot(z,y)
"""