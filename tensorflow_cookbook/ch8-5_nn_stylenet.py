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
style_image  = misc.imresize(style_image, target_shape[1] / style_image.shape[1]) 	#scipy.misc


'''
>>> vgg_path = '/home/ubuntu/Program/datasets/imagenet-vgg-verydeep-19.mat'
>>> original_image_file = 'tensorflow_cookbook/book_cover.jpg'
>>> style_image_file = 'tensorflow_cookbook/starry_night.jpg'
>>> original_image = misc.imread(original_image_file)
>>> style_image = misc.imread(style_image_file)
>>> style_image
array([[[ 69,  59,  49], [ 13,  18,  38], [ 27,  29,  50], ..., [173, 151, 112], [190, 166, 128], [184, 163, 134]],
       [[ 62,  58,  75], [ 25,  33,  79], [ 34,  35,  91], ..., [180, 162, 126], [168, 148, 111], [181, 160, 129]],
       [[ 45,  44,  62], [ 25,  33,  79], [ 32,  32,  82], ..., [201, 182, 150], [189, 171, 135], [172, 148, 114]],
       ..., 
       [[140, 110,  72], [116,  96,  63], [144, 135,  94], ..., [ 73,  75,  54], [100,  92,  55], [141, 124,  72]],
       [[137, 117,  92], [132, 121,  91], [153, 142,  88], ..., [104,  93,  65], [163, 141, 104], [179, 154, 114]],
       [[164, 150, 105], [139, 135, 110], [118, 114,  77], ..., [192, 174, 138], [182, 158, 124], [186, 160, 127]]], dtype=uint8)
>>> style_image.shape
(507, 640, 3)
>>> original_image.shape
(326, 458, 3)
>>> 
>>> style_image.shape[1]
640
>>> target_shape = original_image.shape
>>> target_shape[1]
458
>>> target_shape[1] / style_image.shape[1]
0.715625
>>> style_image = misc.imresize(style_image, target_shape[1] / style_image.shape[1])
>>> style_image.shape
(362, 458, 3)
>>> style_image
array([[[ 50,  47,  55], [ 24,  27,  59], [ 31,  35,  62], ..., [173, 156, 127], [179, 157, 119], [183, 161, 129]],
       [[ 43,  44,  72], [ 29,  33,  83], [ 33,  39,  92], ..., [185, 167, 139], [187, 168, 132], [178, 156, 122]],
       [[ 33,  33,  57], [ 26,  27,  63], [ 24,  30,  60], ..., [179, 162, 130], [191, 174, 144], [184, 164, 131]],
       ..., 
       [[103,  85,  55], [ 66,  69,  56], [ 47,  54,  49], ..., [102,  96,  83], [ 94,  91,  70], [121, 109,  77]],
       [[133, 111,  79], [137, 125,  84], [135, 122,  86], ..., [ 72,  72,  58], [105,  97,  66], [148, 129,  86]],
       [[149, 136, 102], [133, 126,  91], [145, 126,  85], ..., [130, 119,  89], [169, 149, 115], [181, 156, 121]]], dtype=uint8)
>>> 
'''



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


def extract_net_info(path_to_params):
	vgg_data = io.loadmat(path_to_params) 	#scipy.io
	normalization_matrix = vgg_data['normalization'][0][0][0] 	#shape=(224, 224, 3)
	mat_mean = np.mean(normalization_matrix, axis=(0, 1)) 		#shape=(3,) 	mat_mean=array([ 123.68 ,  116.779,  103.939])
	network_weights = vgg_data['layers'][0] 					#shape=(43,)
	return(mat_mean, network_weights)


'''
>>> vgg_data['normalization']
array([[ ( array([[[ 123.68 ,  116.779,  103.939],
				   [ 123.68 ,  116.779,  103.939],
				   [ 123.68 ,  116.779,  103.939],
				   ..., 
				   [ 123.68 ,  116.779,  103.939],
				   [ 123.68 ,  116.779,  103.939],
				   [ 123.68 ,  116.779,  103.939]],
				   [[],...],...]), 
		   array([[ 1.]]), 
		   array([[ 0.,  0.]]), 
		   array([[ 224.,  224.,    3.]]), 
		   array([u'bilinear'], dtype='<U8')
		 )]], dtype=[('averageImage', 'O'), ('keepAspect', 'O'), ('border', 'O'), ('imageSize', 'O'), ('interpolation', 'O')])
>>> 
>>> vgg_data['normalization'][0]
array([( array([[[],...],...]), 
		 array([[ 1.]]), 
		 array([[ 0.,  0.]]), 
		 array([[ 224.,  224.,    3.]]), 
		 array([u'bilinear'], dtype='<U8')
	   )], dtype=[('averageImage', 'O'), ('keepAspect', 'O'), ('border', 'O'), ('imageSize', 'O'), ('interpolation', 'O')])
>>> vgg_data['normalization'][0][0]
( array([[[],...],...]), 
  array([[ 1.]]), 
  array([[ 0.,  0.]]), 
  array([[ 224.,  224.,    3.]]), 
  array([u'bilinear'], dtype='<U8'))
>>> vgg_data['normalization'][0][0][0]
array([[[ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        ..., 
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939]],

       [[ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        ..., 
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939]],

       [[ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        ..., 
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939]],

       ..., 
       [[ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        ..., 
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939]],

       [[ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        ..., 
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939]],

       [[ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        ..., 
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939],
        [ 123.68 ,  116.779,  103.939]]])
>>> 
>>> vgg_data['normalization'][0][0][0].shape
(224, 224, 3)
>>> normalization_matrix = vgg_data['normalization'][0][0][0]
>>> mat_mean = np.mean(normalization_matrix, axis=(0,1))
>>> mat_mean.shape
(3,)
>>> mat_mean
array([ 123.68 ,  116.779,  103.939])
>>> 




>>> vgg_data = io.load(vgg_path)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'load'
>>> vgg_data = io.loadmat(vgg_path)
>>> vgg_data
{'layers': array([[ array([[ (array([[ array([[[[],...],...],...], dtype=float32),
									   array([[]], 				   dtype=float32)]], 	 dtype=object), 
							  array([[]]), 
							  array([u'conv'   ], dtype='<U4'), 
							  array([u'conv1_1'], dtype='<U7'), 
							  array([[]])
							  )]], 
						   dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'], 		dtype='<U4'), 
							  array([u'relu1_1'],	dtype='<U7'))]],
						  dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], 						   dtype=float32)]], dtype=object), 
							  array([[]]), 
							  array([u'conv'], dtype='<U4'), 
							  array([u'conv1_2'], dtype='<U7'), 
							  array([[]]) 
							 )]], 
						   dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'   ], dtype='<U4'), 
							  array([u'relu1_2'], dtype='<U7')
							 )]], 
						  dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([u'pool1'], dtype='<U5'), 
							  array([[]]), 
							  array([[]]), 
							  array([u'pool'], dtype='<U4'), 
							  array([u'max' ], dtype='<U3'), 
							  array([[]])
							 )]], 
						  dtype=[('name', 'O'), ('stride', 'O'), ('pad', 'O'), ('type', 'O'), ('method', 'O'), ('pool', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
									   array([[]], 				   dtype=float32)
									]], dtype=object), 
							  array([[]]), 
							  array([u'conv'   ], dtype='<U4'), 
							  array([u'conv2_1'], dtype='<U7'), 
							  array([[]])
							 )]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'   ], dtype='<U4'), 
							  array([u'relu2_1'], dtype='<U7')
							 )]], dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
									   array([[]], 				   dtype=float32)
									]], dtype=object), 
							  array([[]]), 
							  array([u'conv'   ], dtype='<U4'), 
							  array([u'conv2_2'], dtype='<U7'), 
							  array([[]])
							 )]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'   ], dtype='<U4'), 
							  array([u'relu2_2'], dtype='<U7')
							 )]], dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([u'pool2'], dtype='<U5'), 
							  array([[]]), 
							  array([[]]), 
							  array([u'pool'], dtype='<U4'), 
							  array([u'max'], dtype='<U3'), 
							  array([[]])
							 )]], 
						  dtype=[('name', 'O'), ('stride', 'O'), ('pad', 'O'), ('type', 'O'), ('method', 'O'), ('pool', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
									   array([[]], dtype=float32)
									]], dtype=object), 
							  array([[]]), 
							  array([u'conv'   ], dtype='<U4'), 
							  array([u'conv3_1'], dtype='<U7'), 
							  array([[]])
							 )]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'   ], dtype='<U4'), 
							  array([u'relu3_1'], dtype='<U7')
							 )]], dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
									   array([[]], dtype=float32)
									]], dtype=object), 
							  array([[]]), 
							  array([u'conv'   ], dtype='<U4'), 
							  array([u'conv3_2'], dtype='<U7'), 
							  array([[]])
							 )]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'   ], dtype='<U4'), 
							  array([u'relu3_2'], dtype='<U7')
							 )]], dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
									   array([[]], dtype=float32)
									]], dtype=object), 
							  array([[]]), 
							  array([u'conv'], dtype='<U4'), 
							  array([u'conv3_3'], dtype='<U7'), 
							  array([[]])
							 )]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
					array([[ (array([u'relu'   ], dtype='<U4'), 
							  array([u'relu3_3'], dtype='<U7')
							 )]], dtype=[('type', 'O'), ('name', 'O')]),
					array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						  ]], dtype=object), 
					array([[]]), 
					array([u'conv'   ], dtype='<U4'), 
					array([u'conv3_4'], dtype='<U7'), 
					array([[]])
				 )]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'   ], dtype='<U4'), 
					 array([u'relu3_4'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([u'pool3'], dtype='<U5'), 
					 array([[]]), 
					 array([[]]), 
					 array([u'pool'], dtype='<U4'), 
					 array([u'max'], dtype='<U3'), 
					 array([[]])
					)]], dtype=[('name', 'O'), ('stride', 'O'), ('pad', 'O'), ('type', 'O'), ('method', 'O'), ('pool', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'   ], dtype='<U4'), 
					 array([u'conv4_1'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'   ], dtype='<U4'), 
					 array([u'relu4_1'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'   ], dtype='<U4'), 
					 array([u'conv4_2'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'   ], dtype='<U4'), 
					 array([u'relu4_2'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'   ], dtype='<U4'), 
					 array([u'conv4_3'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'], dtype='<U4'), 
					 array([u'relu4_3'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'   ], dtype='<U4'), 
					 array([u'conv4_4'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'], dtype='<U4'), 
					 array([u'relu4_4'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([u'pool4'], dtype='<U5'), 
					 array([[]]), 
					 array([[]]), 
					 array([u'pool'], dtype='<U4'), 
					 array([u'max'], dtype='<U3'), 
					 array([[]])
					)]], dtype=[('name', 'O'), ('stride', 'O'), ('pad', 'O'), ('type', 'O'), ('method', 'O'), ('pool', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'], dtype='<U4'), 
					 array([u'conv5_1'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'   ], dtype='<U4'), 
					 array([u'relu5_1'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'], dtype='<U4'), 
					 array([u'conv5_2'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'], dtype='<U4'), 
					 array([u'relu5_2'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'], dtype='<U4'), 
					 array([u'conv5_3'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'], dtype='<U4'), 
					 array([u'relu5_3'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]), 
		   array([[ (array([[ array([[[[],...],...],...], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'   ], dtype='<U4'), 
					 array([u'conv5_4'], dtype='<U7'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu'], dtype='<U4'), 
					 array([u'relu5_4'], dtype='<U7')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([u'pool5'], dtype='<U5'), 
					 array([[]]), 
					 array([[]]), 
					 array([u'pool'], dtype='<U4'), 
					 array([u'max'], dtype='<U3'), 
					 array([[]])
					)]], dtype=[('name', 'O'), ('stride', 'O'), ('pad', 'O'), ('type', 'O'), ('method', 'O'), ('pool', 'O')]),
		   array([[ (array([[ array([[[[]]]], dtype=float32),
							  array([[]], dtype=float32)
						   ]], dtype=object), 
					 array([[]]), 
					 array([u'conv'], dtype='<U4'), 
					 array([u'fc6'], dtype='<U3'), 
					 array([[]])
					)]], dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')]),
		   array([[ (array([u'relu' ], dtype='<U4'), 
					 array([u'relu6'], dtype='<U5')
					)]], dtype=[('type', 'O'), ('name', 'O')]),
		   array([[ (array([[ array([[[[
'''


# Create the VGG-19 Network
def vgg_network(network_weights, init_image):
	network = {}
	image = init_image
	print("init_image\t", image)

	for i, layer in enumerate(vgg_layers):
		#if layer[1] == 'c':
		if layer[0] == 'c':
			weights, bias = network_weights[i][0][0][0][0]
			weights = np.transpose(weights, (1, 0, 2, 3))
			bias    = bias.reshape(-1)
			conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
			image = tf.nn.bias_add(conv_layer, bias)
			print('\t c:\t', weights.shape, bias.shape, conv_layer, image)
		#elif layer[1] == 'r':
		elif layer[0] == 'r':
			image = tf.nn.relu(image)
			print('\t r \t', image)
		else:
			image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
			print('\t e \t', image)
		network[layer] = image
		print("i=", i, 'layer=', layer, 'image=', image)
	return(network)


# Here we define which layers apply to the original or style image
original_layer = 'relu4_2'
style_layers   = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# Get network parameters
normalization_mean, network_weights = extract_net_info(vgg_path)

shape = (1,) + original_image.shape
style_shape = (1,) + style_image.shape
original_features = {}
style_features    = {}

# Get network parameter
image = tf.placeholder('float', shape=shape)
vgg_net = vgg_network(network_weights, image)





'''
>>> weights, bias = network_weights[ 0 ][0][0][0][0]
>>> weights.shape
(3, 3, 3, 64)
>>> bias.shape
(1, 64)
>>> len(network_weights)
43
>>> 
>>> original_image.shape
(326, 458, 3)
>>> (1,) + original_image.shape
(1, 326, 458, 3)
>>> (1,) + style_image.shape
(1, 326, 458, 3)
>>> 

>>> tf.placeholder('float', shape=(2,3))
<tf.Tensor 'Placeholder:0' shape=(2,3) dtype=float32>

>>> a = tf.placeholder('float', shape=(2,3))
>>> print a
Tensor("Placeholder_1:0", shape=(2, 3), dtype=float32)
>>> a
<tf.Tensor 'Placeholder_1:0' shape=(2, 3) dtype=float32>
>>> 




2018-03-05 18:56:42.052099: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
 					init_image	Tensor("Placeholder:0", shape=(1, 326, 458, 3), dtype=float32)
i=  0 layer= conv1_1 	image= Tensor("BiasAdd:0", 		shape=(1, 326, 458, 64), dtype=float32)
i=  1 layer= relu1_1 	image= Tensor("Relu:0", 		shape=(1, 326, 458, 64), dtype=float32)
i=  2 layer= conv1_2 	image= Tensor("BiasAdd_1:0", 	shape=(1, 326, 458, 64), dtype=float32)
i=  3 layer= relu1_2 	image= Tensor("Relu_1:0", 		shape=(1, 326, 458, 64), dtype=float32)
i=  4 layer= pool1 		image= Tensor("MaxPool:0", 		shape=(1, 163, 229, 64), dtype=float32)
i=  5 layer= conv2_1 	image= Tensor("BiasAdd_2:0", 	shape=(1, 163, 229, 128), dtype=float32)
i=  6 layer= relu2_1 	image= Tensor("Relu_2:0", 		shape=(1, 163, 229, 128), dtype=float32)
i=  7 layer= conv2_2 	image= Tensor("BiasAdd_3:0", 	shape=(1, 163, 229, 128), dtype=float32)
i=  8 layer= relu2_2 	image= Tensor("Relu_3:0", 		shape=(1, 163, 229, 128), dtype=float32)
i=  9 layer= pool2 		image= Tensor("MaxPool_1:0", 	shape=(1, 82, 115, 128), dtype=float32)
i= 10 layer= conv3_1 	image= Tensor("BiasAdd_4:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 11 layer= relu3_1 	image= Tensor("Relu_4:0", 		shape=(1, 82, 115, 256), dtype=float32)
i= 12 layer= conv3_2 	image= Tensor("BiasAdd_5:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 13 layer= relu3_2 	image= Tensor("Relu_5:0", 		shape=(1, 82, 115, 256), dtype=float32)
i= 14 layer= conv3_3 	image= Tensor("BiasAdd_6:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 15 layer= relu3_3 	image= Tensor("Relu_6:0", 		shape=(1, 82, 115, 256), dtype=float32)
i= 16 layer= conv3_4 	image= Tensor("BiasAdd_7:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 17 layer= relu3_4 	image= Tensor("Relu_7:0", 		shape=(1, 82, 115, 256), dtype=float32)
i= 18 layer= pool3 		image= Tensor("MaxPool_2:0", 	shape=(1, 41, 58, 256), dtype=float32)
i= 19 layer= conv4_1 	image= Tensor("BiasAdd_8:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 20 layer= relu4_1 	image= Tensor("Relu_8:0", 		shape=(1, 41, 58, 512), dtype=float32)
i= 21 layer= conv4_2 	image= Tensor("BiasAdd_9:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 22 layer= relu4_2 	image= Tensor("Relu_9:0", 		shape=(1, 41, 58, 512), dtype=float32)
i= 23 layer= conv4_3 	image= Tensor("BiasAdd_10:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 24 layer= relu4_3 	image= Tensor("Relu_10:0", 		shape=(1, 41, 58, 512), dtype=float32)
i= 25 layer= conv4_4 	image= Tensor("BiasAdd_11:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 26 layer= relu4_4 	image= Tensor("Relu_11:0", 		shape=(1, 41, 58, 512), dtype=float32)
i= 27 layer= pool4 		image= Tensor("MaxPool_3:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 28 layer= conv5_1 	image= Tensor("BiasAdd_12:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 29 layer= relu5_1 	image= Tensor("Relu_12:0", 		shape=(1, 21, 29, 512), dtype=float32)
i= 30 layer= conv5_2 	image= Tensor("BiasAdd_13:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 31 layer= relu5_2 	image= Tensor("Relu_13:0", 		shape=(1, 21, 29, 512), dtype=float32)
i= 32 layer= conv5_3 	image= Tensor("BiasAdd_14:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 33 layer= relu5_3 	image= Tensor("Relu_14:0", 		shape=(1, 21, 29, 512), dtype=float32)
i= 34 layer= conv5_4 	image= Tensor("BiasAdd_15:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 35 layer= relu5_4 	image= Tensor("Relu_15:0", 		shape=(1, 21, 29, 512), dtype=float32)
[Finished in 76.3s]




2018-03-05 19:30:30.938250: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
init_image	 Tensor("Placeholder:0", shape=(1, 326, 458, 3), dtype=float32)
						 c:	 (3, 3, 3, 64) (64,) 
							 Tensor("Conv2D:0",  	shape=(1, 326, 458, 64), dtype=float32) 
							 Tensor("BiasAdd:0", 	shape=(1, 326, 458, 64), dtype=float32)
i=  0 layer= conv1_1 image=  Tensor("BiasAdd:0", 	shape=(1, 326, 458, 64), dtype=float32)
						 r 	 Tensor("Relu:0", 		shape=(1, 326, 458, 64), dtype=float32)
i=  1 layer= relu1_1 image=  Tensor("Relu:0", 		shape=(1, 326, 458, 64), dtype=float32)
						 c:	 (3, 3, 64, 64) (64,) 
							 Tensor("Conv2D_1:0", 	shape=(1, 326, 458, 64), dtype=float32) 
							 Tensor("BiasAdd_1:0", 	shape=(1, 326, 458, 64), dtype=float32)
i=  2 layer= conv1_2 image=  Tensor("BiasAdd_1:0", 	shape=(1, 326, 458, 64), dtype=float32)
						 r 	 Tensor("Relu_1:0", 	shape=(1, 326, 458, 64), dtype=float32)
i=  3 layer= relu1_2 image=  Tensor("Relu_1:0", 	shape=(1, 326, 458, 64), dtype=float32)
						 e 	 Tensor("MaxPool:0", 	shape=(1, 163, 229, 64), dtype=float32)
i=  4 layer= pool1   image=  Tensor("MaxPool:0", 	shape=(1, 163, 229, 64), dtype=float32)
						 c:	 (3, 3, 64, 128) (128,) 
							 Tensor("Conv2D_2:0", 	shape=(1, 163, 229, 128), dtype=float32) 
							 Tensor("BiasAdd_2:0", 	shape=(1, 163, 229, 128), dtype=float32)
i=  5 layer= conv2_1 image=  Tensor("BiasAdd_2:0", 	shape=(1, 163, 229, 128), dtype=float32)
						 r 	 Tensor("Relu_2:0", 	shape=(1, 163, 229, 128), dtype=float32)
i=  6 layer= relu2_1 image=  Tensor("Relu_2:0", 	shape=(1, 163, 229, 128), dtype=float32)
						 c:	 (3, 3, 128, 128) (128,) 
							 Tensor("Conv2D_3:0", 	shape=(1, 163, 229, 128), dtype=float32) 
							 Tensor("BiasAdd_3:0", 	shape=(1, 163, 229, 128), dtype=float32)
i=  7 layer= conv2_2 image=  Tensor("BiasAdd_3:0", 	shape=(1, 163, 229, 128), dtype=float32)
						 r 	 Tensor("Relu_3:0", 	shape=(1, 163, 229, 128), dtype=float32)
i=  8 layer= relu2_2 image=  Tensor("Relu_3:0", 	shape=(1, 163, 229, 128), dtype=float32)
						 e 	 Tensor("MaxPool_1:0", 	shape=(1, 82, 115, 128), dtype=float32)
i=  9 layer= pool2   image=  Tensor("MaxPool_1:0", 	shape=(1, 82, 115, 128), dtype=float32)
						 c:	 (3, 3, 128, 256) (256,) 
							 Tensor("Conv2D_4:0", 	shape=(1, 82, 115, 256), dtype=float32) 
							 Tensor("BiasAdd_4:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 10 layer= conv3_1 image=  Tensor("BiasAdd_4:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 r 	 Tensor("Relu_4:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 11 layer= relu3_1 image=  Tensor("Relu_4:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 c:	 (3, 3, 256, 256) (256,) 
							 Tensor("Conv2D_5:0", 	shape=(1, 82, 115, 256), dtype=float32) 
							 Tensor("BiasAdd_5:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 12 layer= conv3_2 image=  Tensor("BiasAdd_5:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 r 	 Tensor("Relu_5:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 13 layer= relu3_2 image=  Tensor("Relu_5:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 c:	 (3, 3, 256, 256) (256,) 
							 Tensor("Conv2D_6:0", 	shape=(1, 82, 115, 256), dtype=float32) 
							 Tensor("BiasAdd_6:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 14 layer= conv3_3 image=  Tensor("BiasAdd_6:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 r 	 Tensor("Relu_6:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 15 layer= relu3_3 image=  Tensor("Relu_6:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 c:	 (3, 3, 256, 256) (256,) 
							 Tensor("Conv2D_7:0", 	shape=(1, 82, 115, 256), dtype=float32) 
							 Tensor("BiasAdd_7:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 16 layer= conv3_4 image=  Tensor("BiasAdd_7:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 r 	 Tensor("Relu_7:0", 	shape=(1, 82, 115, 256), dtype=float32)
i= 17 layer= relu3_4 image=  Tensor("Relu_7:0", 	shape=(1, 82, 115, 256), dtype=float32)
						 e 	 Tensor("MaxPool_2:0", 	shape=(1, 41, 58, 256), dtype=float32)
i= 18 layer= pool3   image=  Tensor("MaxPool_2:0", 	shape=(1, 41, 58, 256), dtype=float32)
						 c:	 (3, 3, 256, 512) (512,) 
							 Tensor("Conv2D_8:0", 	shape=(1, 41, 58, 512), dtype=float32) 
							 Tensor("BiasAdd_8:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 19 layer= conv4_1 image=  Tensor("BiasAdd_8:0", 	shape=(1, 41, 58, 512), dtype=float32)
						 r 	 Tensor("Relu_8:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 20 layer= relu4_1 image=  Tensor("Relu_8:0", 	shape=(1, 41, 58, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_9:0", 	shape=(1, 41, 58, 512), dtype=float32) 
							 Tensor("BiasAdd_9:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 21 layer= conv4_2 image=  Tensor("BiasAdd_9:0", 	shape=(1, 41, 58, 512), dtype=float32)
						 r 	 Tensor("Relu_9:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 22 layer= relu4_2 image=  Tensor("Relu_9:0", 	shape=(1, 41, 58, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_10:0", 	shape=(1, 41, 58, 512), dtype=float32) 
							 Tensor("BiasAdd_10:0", shape=(1, 41, 58, 512), dtype=float32)
i= 23 layer= conv4_3 image=  Tensor("BiasAdd_10:0", shape=(1, 41, 58, 512), dtype=float32)
						 r 	 Tensor("Relu_10:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 24 layer= relu4_3 image=  Tensor("Relu_10:0", 	shape=(1, 41, 58, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_11:0", 	shape=(1, 41, 58, 512), dtype=float32) 
							 Tensor("BiasAdd_11:0", shape=(1, 41, 58, 512), dtype=float32)
i= 25 layer= conv4_4 image=  Tensor("BiasAdd_11:0", shape=(1, 41, 58, 512), dtype=float32)
						 r 	 Tensor("Relu_11:0", 	shape=(1, 41, 58, 512), dtype=float32)
i= 26 layer= relu4_4 image=  Tensor("Relu_11:0", 	shape=(1, 41, 58, 512), dtype=float32)
						 e 	 Tensor("MaxPool_3:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 27 layer= pool4   image=  Tensor("MaxPool_3:0", 	shape=(1, 21, 29, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_12:0", 	shape=(1, 21, 29, 512), dtype=float32) 
							 Tensor("BiasAdd_12:0", shape=(1, 21, 29, 512), dtype=float32)
i= 28 layer= conv5_1 image=  Tensor("BiasAdd_12:0", shape=(1, 21, 29, 512), dtype=float32)
						 r 	 Tensor("Relu_12:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 29 layer= relu5_1 image=  Tensor("Relu_12:0", 	shape=(1, 21, 29, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_13:0", 	shape=(1, 21, 29, 512), dtype=float32) 
							 Tensor("BiasAdd_13:0", shape=(1, 21, 29, 512), dtype=float32)
i= 30 layer= conv5_2 image=  Tensor("BiasAdd_13:0", shape=(1, 21, 29, 512), dtype=float32)
						 r 	 Tensor("Relu_13:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 31 layer= relu5_2 image=  Tensor("Relu_13:0", 	shape=(1, 21, 29, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_14:0", 	shape=(1, 21, 29, 512), dtype=float32) 
							 Tensor("BiasAdd_14:0", shape=(1, 21, 29, 512), dtype=float32)
i= 32 layer= conv5_3 image=  Tensor("BiasAdd_14:0", shape=(1, 21, 29, 512), dtype=float32)
						 r 	 Tensor("Relu_14:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 33 layer= relu5_3 image=  Tensor("Relu_14:0", 	shape=(1, 21, 29, 512), dtype=float32)
						 c:	 (3, 3, 512, 512) (512,) 
							 Tensor("Conv2D_15:0", 	shape=(1, 21, 29, 512), dtype=float32) 
							 Tensor("BiasAdd_15:0", shape=(1, 21, 29, 512), dtype=float32)
i= 34 layer= conv5_4 image=  Tensor("BiasAdd_15:0", shape=(1, 21, 29, 512), dtype=float32)
						 r 	 Tensor("Relu_15:0", 	shape=(1, 21, 29, 512), dtype=float32)
i= 35 layer= relu5_4 image=  Tensor("Relu_15:0", 	shape=(1, 21, 29, 512), dtype=float32)
[Finished in 41.7s]


'''










'''
references:
https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/05_Stylenet_NeuralStyle
'''

