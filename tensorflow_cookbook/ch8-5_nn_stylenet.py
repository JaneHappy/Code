# coding: utf-8

from __future__ import division
from __future__ import print_function

import os
from scipy import misc 	##import scipy.misc
from scipy import io 	##import scipy.io 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops 

ops.reset_default_graph()
sess = tf.Session()


# Image Files
original_image_file = 'book_cover.jpg'   	##'temp/book_cover.jpg'
style_image_file    = 'starry_night.jpg' 	##'temp/starry_night.jpg'

# Saved VGG Network path
vgg_path = '/home/ubuntu/Program/datasets/imagenet-vgg-verydeep-19.mat'  ##'/home/nick/Documents/tensorflow/vgg_19_models/imagenet-vgg-verydeep-19.mat'

# Default Arguments
original_image_weight = 5.0
style_image_weight    = 500.0
regularization_weight = 100
learning_rate 		  = 0.001
generations 		  = 5000
output_generations    = 250
beta1 				  = 0.9 	# For the Adam optimizer
beta2 				  = 0.999 	# For the Adam optimizer

# Read in images
original_image = misc.imread(original_image_file) 	##scipy.misc.imread(original_image_file)
style_image    = misc.imread(style_image_file) 		##scipy.misc.imread(style_image_file)

# Get shape of target and make the style image the same
target_shape = original_image.shape
style_image  = misc.imresize(style_image, target[1] / style_image.shape[1]) 	#scipy.misc

# VGG-19 Layer Setup (From paper)
vgg_layers = ['conv1_1', 'relu1_1', 
			  'conv1_2', 'relu1_2', 'pool1', 
			  'conv2_1', 'relu2_1', 
			  'conv2_2', 'relu2_2', 'pool2',
			  'conv3_1', 'relu3_1', 
			  'conv3_2', 'relu3_2', 
			  'conv3_3', 'relu3_3', 
			  'conv3_4', 'relu3_4', 'pool3', 
			  'conv4_1', 'relu4_1', 
			  'conv4_2', 'relu4_2', 
			  'conv4_3', 'relu4_3', 
			  'conv4_4', 'relu4_4', 'pool4', 
			  'conv5_1', 'relu5_1', 
			  'conv5_2', 'relu5_2', 
			  'conv5_3', 'relu5_3', 
			  'conv5_4', 'relu5_4']












'''
references:
https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/05_Stylenet_NeuralStyle
'''

