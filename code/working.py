# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:32:15 2019

@author: Raghav N G
"""



import tflearn
import tensorflow as tf

import pandas as pd
import numpy as np


"""
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
 
Y = [
    [0],  # Desired output for inputs 0, 0
    [1],  # Desired output for inputs 0, 1
    [1],  # Desired output for inputs 1, 0
    [0]   # Desired output for inputs 1, 1
]
"""



#==========
df = pd.read_csv('pee.csv')

inp =df.to_numpy()

final_inp =[]
final_inp=inp[:,0:4]
output=inp[:,4]

final_output=[]

for i in output:
    if i=="Cancer":
        final_output.append([1,0,0,0])
        
    if i=="Diabeties":
        final_output.append([0,1,0,0])
        
        
    if i=="Stomach":
        final_output.append([0,0,1,0])
        
        
        
    if i=="Heart":
        final_output.append([0,0,0,1])  
        
        
        
final_output = np.array(final_output) 



#=======


weights = tflearn.initializations.uniform(minval = -1, maxval = 1)
tf.reset_default_graph()
# Input layer
net = tflearn.input_data(
        shape = [None, 4],
        name = 'my_input'
)
 
# Hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
 
# Output layer
net = tflearn.fully_connected(net, 4,
        activation = 'softmax', 
        name = 'my_output'
)

net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load('xor.tflearn')
except:
    model = tflearn.DNN(net)
    model.fit(final_inp, final_output, 250)

results = model.predict([[40,40,70,30]])
results_index = np.argmax(results)
print(results_index)
# Remove train ops
with net.graph.as_default():
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
 
# Save the model
model.save('xor.tflearn')

food={0:[["maidha","Coliflower","msdklsfngskl"]],1:[["maidha","Coliflower","msdklsfngskl"]],2:[["maidha","Coliflower","msdklsfngskl"]],3:[["maidha","Coliflower","msdklsfngskl"]]}  

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