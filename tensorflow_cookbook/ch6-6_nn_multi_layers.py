# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/06_Using_Multiple_Layers

from __future__ import division
from __future__ import print_function


'''
# Using a Multiple Layer Network

We will illustrate how to use a Multiple Layer Network in TensorFlow

Low Birthrate data:
	#Columns    Variable                                      Abbreviation
	#---------------------------------------------------------------------
	# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
	#                          1 = Birth Weight < 2500g)
	# Age of the Mother in Years                              AGE
	# Weight in Pounds at the Last Menstrual Period           LWT
	# Race (1 = White, 2 = Black, 3 = Other)                  RACE
	# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
	# History of Premature Labor (0 = None  1 = One, etc.)    PTL
	# History of Hypertension (1 = Yes, 0 = No)               HT
	# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
	# Birth Weight in Grams                                   BWT
	#---------------------------------------------------------------------
The multiple neural network layer we will create will be composed of three fully connected hidden layers, with node sizes 50, 25, and 5
'''

import csv
import os
import os.path
import random
import requests

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops 




## Obtain the data



















