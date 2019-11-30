# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 08:28:21 2019

@author: Raghav N G
"""

import tensorflow as tf
from tensorflow import lite

graph_def_file = 'frozen_model.pb'

input_arrays = ["my_input/X"]
output_arrays = ["my_output/Softmax"]

converter = tf.lite.TocoConverter.from_frozen_graph(
        graph_def_file,input_arrays,output_arrays)

tflite_model = converter.convert()
open("converted_model.tflite","wb").write(tflite_model)
