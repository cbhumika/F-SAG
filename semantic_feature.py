# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:37:49 2019

@author: acer
"""
#import sklearn
# Import all of the scikit learn stuff
#from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#from scipy.sparse.coo_matrix 
#from sklearn import metrics
#from sklearn.cluster import KMeans, MiniBatchKMeans

def lsa(s1,s2,d):
   #s3=s2.iloc[0].to_string()
   #s4=s2.iloc[1].to_string()
   #print(s1)
   #print(s1)
   #example = [s1]
  # print(s2)
   #example.append(s3)
   #example.append(s4)
   #print(s2)
   """ for i in range(len(s2)) : 
       #print(s2.iloc[i])
       l=s2.iloc[i].to_string()
       #print(l)
       example.append(l)
   
   #print(example)"""
   #print(len(s2))
   dfToList = s2['ass'].tolist()
   #print(len(dfToList))
   dfToList.insert(0,s1)
   #print(dfToList)
   
   vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
   dtm = vectorizer.fit_transform(dfToList)
   dtm=dtm.asfptype()
#print(pd.DataFrame(dtm.toarray(),index=example,columns=vectorizer.get_feature_names()).head(10))

# Get words that correspond to each column
#print(vectorizer.get_feature_names())

# Fit LSA. Use algorithm = “randomized” for large datasets
   lsa = TruncatedSVD(2, algorithm = 'arpack')
   dtm_lsa = lsa.fit_transform(dtm)
   dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
#print(pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns =vectorizer.get_feature_names()))
#print(pd.DataFrame(dtm_lsa, index = example, columns = ["component_1","component_2"]))

#xs = [w[0] for w in dtm_lsa]
#ys = [w[1] for w in dtm_lsa]
#print(xs, ys)
#figure()
   """plt.scatter(xs,ys)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Plot of points against LSA principal components')
plt.show()

plt.figure()
ax = plt.gca()
ax.quiver(0,0,xs,ys,angles='xy',scale_units='xy',scale=1, linewidth = .01)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Plot of points against LSA principal components')
plt.draw()
plt.show()"""

# Compute document similarity using LSA components
   res=pd.DataFrame()
   similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
   #print(similarity)
   res = pd.DataFrame(similarity)[0]
   #print(res)
   """df=pd.DataFrame()
   df.insert(0, "lsa1",res)
   df.to_csv('y%d.csv'%d)"""
  
   return res 









