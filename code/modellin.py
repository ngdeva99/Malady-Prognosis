# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:01:41 2019

@author: Raghav N G
"""

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
import pandas as pd

df = pd.read_csv('pee.csv')

#print(df)

inp = df.to_numpy()
final_inp = inp[:,0:4]
#print(final_inp)

output = inp[:,4]
#print(output)
final_output = []

for i in output:
    if i=="Cancer":
        final_output.append([0,0,0,1])
    elif i=="Heart":
        final_output.append([1,0,0,0])
    elif i=="Stomach":
        final_output.append([0,1,0,0])
    elif i=="Diabeties":
        final_output.append([0,0,1,0])        
    

final_array = np.array(final_output)

rbc = np.array(df['RBC'])
#print(rbc)
sugar = np.array(df['Sugar'])

acidity = np.array(df['Acidity'])

colestrol=np.array(df['Colestrol'])


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 4, activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
ch=1
if ch!=1:
    model.load("model.tflearn")
else:
    model.fit(final_inp, final_output, n_epoch=300, batch_size=1, show_metric=True)
    model.save("model.tflearn")
    
results = model.predict([[40,40,70,30]])
results_index = np.argmax(results)    
print(results_index)

food={0:[["to avoid","Coliflower","alcohol"]],1:[["avoid","icecream","oily foods"]],2:[["avoid","aerated drinks","fried items"]],3:[["avoid","ghee substances","junk foods"]]}  

if results_index==0:
    print("Cancer")
    print(food[0])
        
if results_index==1:
    print("Diabeties")
    print(food[1])
        
        
if results_index==2:
    print("Stomach")
    print(food[2])
        
        
if results_index==3:
    print("Heart")
    print(food[3])