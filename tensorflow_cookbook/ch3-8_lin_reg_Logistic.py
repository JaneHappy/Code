# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/08_Implementing_Logistic_Regression

from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt 
import numpy as np 

import requests
import os.path
import csv

import tensorflow as tf 
from tensorflow.python.framework import ops


ops.reset_default_graph()
#	 Create graph
sess = tf.Session()

## Obtain and prepare data for modeling
#	 name of data file
birth_weight_file = 'birth_weight.csv'

# download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
	birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/' + \
		'raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
	birth_file   = requests.get(birthdata_url)
	birth_data   = birth_file.text.split('\r\n')
	birth_header = birth_data[0].split('\t')
	birth_data   = [ [ float(x)  for x in y.split('\t') if len(x)>=1]  for y in birth_data[1:] if len(y)>=1]
	with open(birth_weight_file, "w") as f:
		writer = csv.writer(f)
		writer.writerow(birth_header)
		writer.writerows(birth_data)
		f.close()

'''
2018-03-02 01:16:32.435235: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 8.2s]
'''



