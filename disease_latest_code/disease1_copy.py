# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:53:01 2021

@author: PS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset1 = pd.read_csv('Diseases_test_set.csv')
y_test = dataset1.iloc[::,0].values
X_test = dataset1.iloc[::,1:1868].values

dataset = pd.read_csv('Diseases_train_set.csv')
y_train = dataset.iloc[:,0].values
X_train = dataset.iloc[:,1:1868].values

list1=[]
list2=[]
j=0
for i in range(1,676):
    list1.append(j)
    if(i%5==0):
        j=j+1
for i in range(0,135):
    list2.append(i)
    
y_train=list1
y_test=list2
l1 = pd.DataFrame(y_train)    
y_train= l1.iloc[:,0].values
l2 = pd.DataFrame(y_test)    
y_test= l2.iloc[:,0].values


from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
gnb=gnb.fit(X_train,(y_train))
from sklearn.metrics import accuracy_score
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))

import pickle
with open('symptoms_final.pkl', 'rb') as f:
    symptoms = pickle.load(f)

with open('diseases_final.pkl', 'rb') as ff:
    diseases = pickle.load(ff)

