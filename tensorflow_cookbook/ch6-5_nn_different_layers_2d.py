# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/06_Neural_Networks/05_Implementing_Different_Layers/05_implementing_different_layers.ipynb

from __future__ import division
from __future__ import print_function


# Implementing Different Layers
import csv
import os
import random
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops 



#	 Reset Graph
ops.reset_default_graph()
sess = tf.Session()

#	 parameters for the run
row_size = 10
col_size = 10
conv_size        = 2
conv_stride_size = 2
maxpool_size        = 2
maxpool_stride_size = 1

#	 ensure reproducibility
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

#	 Generate 2D data
data_size = [row_size, col_size]
data_2d   = np.random.normal(size=data_size)

#--------Placeholder--------
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size) 	#[row_size, col_size]



#	 Convolution
def conv_layer_2d(input_2d, my_filter, stride_size):
	#	 TensorFlow's 'conv2d()' function only works with 4D arrays:
	#	 [batch#, width, height, channels], we have 1 batch, and
	#	 1 channel, but we do have width AND height this time.
	#	 So next we create the 4D array by inserting dimension 1's.
	input_3d = tf.expand_dims(input_2d, 0) 			#[1, row_size,col_size]
	input_4d = tf.expand_dims(input_3d, 3) 			#[1, row_size,col_size, 1]
	#	 Note the stride difference below!
	convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, 
									  strides=[1, stride_size, stride_size, 1], padding='VALID')
	# Get rid of unnecessary dimensions
	conv_output_2d = tf.squeeze(convolution_output)
	return(conv_output_2d)

#	 Create Convolutional Filter
my_filter = tf.Variable(tf.random_normal(shape=[conv_size, conv_size, 1, 1]))
#	 Create Convolutional Layer
my_convolution_output = conv_layer_2d(x_input_2d, my_filter, stride_size=conv_stride_size)

'''
x_input_2d 		#[row_size,col_size]
	input_4d 		#[1, row_size,col_size, 1]
	filter 			#[conv_size, conv_size, 1,1]
	strides 		#[1, conv_stride_size, conv_stride_size, 1]
		(1) filter 			-> 	#[conv_size*conv_size*1, 1]
		(2) input_4d 		-> 	#[1, out_h,out_w, conv_size*conv_size*1]
		(3) patch*filter' 	-> 	#[1, out_h,out_w, 1]
		(i ) out_h 			ceil((row_size-conv_size+1) / conv_stride_size)
		(ii) out_w 			ceil((col_size-conv_size+1) / conv_stride_size)
							sp. ceil((10-2+1)/2) = ceil(9/2) = ceil(4.5) = 5
		convolution_output 		#[1, out_h,out_w, 1]
		conv_output_2d 			#[out_h, out_w]

my_convolution_output 			#[out_h, out_w]  i.e., [5,5]
'''



#--------Activation--------
def activation(input_1d):
	return(tf.nn.relu(input_1d))

#	 Create Activation Layer
my_activation_output = activation(my_convolution_output)

'''
my_activation_output 			#[out_h, out_w]  i.e., [5,5]
'''



#--------Max Pool--------
def max_pool(input_2d, width, height, stride):
	#	 Just like 'conv2d()' above, max_pool() works with 4D arrays.
	#	 [batch_size=1, width=given, height=given, channels=1]
	input_3d = tf.expand_dims(input_2d, 0) 		#[1, out_h,out_w]
	input_4d = tf.expand_dims(input_3d, 3) 		#[1, out_h,out_w, 1]
	#	 Perform the max pooling with strides = [1,1,1,1]
	#	 If we wanted to increase the stride on our data dimension, say by
	#	 a factor of '2', we put strides = [1, 2, 2, 1]
	pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1],
								 strides=[1, stride, stride, 1], padding='VALID')
	#	 Get rid of unnecessary dimensions
	pool_output_2d = tf.squeeze(pool_output)
	return(pool_output_2d)

#	 Create Max-Pool Layer
my_maxpool_output = max_pool(my_activation_output, width=maxpool_size, height=maxpool_size, stride=maxpool_stride_size)

'''
my_activation_output 		#[out_h,out_w] 	i.e., [5,5]
	input_4d 			#[1, out_h,out_w, 1]
	ksize 				#[1, maxpool_size, maxpool_size, 1]
	strides 			#[1, maxpool_stride_size, maxpool_stride_size, 1]
		pool_output 		#[1, (out_h-maxpool_size)/maxpool_stride_size+1, (out_w-maxpool_size)/maxpool_stride_size+1, 1]
		pool_output_2d 		#[(out_h-maxpool_size)/maxpool_stride_size+1, (out_w-maxpool_size)/maxpool_stride_size+1]
							 i.e., (5-2)/1+1=3/1+1=4,  [4,4]
my_maxpool_output 		#[(out_h-maxpool_size)/maxpool_stride_size+1, (out_w-maxpool_size)/maxpool_stride_size+1]  	i.e., [4,4]
'''



#--------Fully Connected--------
def fully_connected(input_layer, num_outputs):
	#	 In order to connect our whole W by H 2d array, we first flatten it out to
	#	 a W times H 1D array.
	flat_input = tf.reshape(input_layer, [-1])				#[row,col]
	'''
	input_layer = [5x5], 
	flat_input    [row1, row2, row3, row4, ...]
	'''
	#	 We then find out how long it is, and create an array for the shape of
	#	 the multiplication weight = (WxH) by (num_outputs)
	weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
	'''
	output:  	[row * col, num_outputs]
				row = 	(out_h-maxpool_size)/maxpool_stride_size+1
				col = 	(out_w-maxpool_size)/maxpool_stride_size+1
	'''
	#	 Initialize the weight
	weight = tf.random_normal(weight_shape, stddev=0.1) 	#[row*col, num_outputs]
	#	 Initialize the bias
	bias   = tf.random_normal(shape=[num_outputs]) 			#[num_outputs,]
	#	 Now make the flat 1D array into a 2D array for multiplication
	input_2d = tf.expand_dims(flat_input, 0) 				#[1, row*col]
	#	 Multiply and add the bias
	full_output = tf.add(tf.matmul(input_2d, weight), bias) 	#[1, num_outputs]
	#	 Get rid of extra dimension
	full_output_2d = tf.squeeze(full_output) 				#[num_outputs,]
	return(full_output_2d)

#	 Create Fully Connected Layer
my_full_output = fully_connected(my_maxpool_output, 5)

'''
my_full_output 		#[num_outputs,] 	i.e., [5,]
'''



# Run graph
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}

print('>>>> 2D Data <<<<')

#	 Convolution Output
print('Input = %s array' % (x_input_2d.shape.as_list()))
print('%s Convolution, stride size = [%d, %d] , results in the %s array' % 
	  (my_filter.get_shape().as_list()[:2], conv_stride_size, conv_stride_size, my_convolution_output.shape.as_list()))
print(sess.run(my_convolution_output, feed_dict=feed_dict))
'''
x_input_2d 				#[row_size, col_size]
my_filter 				#[conv_size, conv_size, 1, 1]
my_convolution_output 	#[out_h, out_w]
		(i ) out_h 			ceil((row_size-conv_size+1) / conv_stride_size)
		(ii) out_w 			ceil((col_size-conv_size+1) / conv_stride_size)
'''

#	 Activation Output
print('\nInput = the above %s array' % (my_convolution_output.shape.as_list()))
print('ReLU element wise returns the %s array' % (my_activation_output.shape.as_list()))
print(sess.run(my_activation_output, feed_dict=feed_dict))
'''
my_activation_output 	#[out_h, out_w] 	i.e., [5,5]
'''

# Max Pool Output
print('\nInput = the above %s array' % (my_activation_output.shape.as_list()))
print('MaxPool, stride size = [%d, %d], results in %s array' % 
	  (maxpool_stride_size, maxpool_stride_size, my_maxpool_output.shape.as_list()))
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
'''
my_maxpool_output 		#[(out_h-maxpool_size)/maxpool_stride_size+1, (out_w-maxpool_size)/maxpool_stride_size+1] 
											i.e., [4,4]
'''

# Fully Connected Output
print('\nInput = the above %s array' % (my_maxpool_output.shape.as_list()))
print('Fully connected layer on all %d rows results in %s outputs:' % 
	  (my_maxpool_output.shape.as_list()[0],my_full_output.shape.as_list()[0]))
print(sess.run(my_full_output, feed_dict=feed_dict))
'''
my_full_output 			#[num_outputs,] 	i.e., [5,]
'''




'''
2018-03-02 13:39:47.522858: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
>>>> 2D Data <<<<
Input = [10, 10] array
[2, 2] Convolution, stride size = [2, 2] , results in the [5, 5] array
[[ 0.14431179  0.72783369  1.51149166 -1.28099763  1.78439188]
 [-2.54503059  0.76156765 -0.51650006  0.77131093  0.37542343]
 [ 0.49345911  0.01592223  0.38653135 -1.47997665  0.6952765 ]
 [-0.34617192 -2.53189754 -0.9525758  -1.4357065   0.66257358]
 [-1.98540258  0.34398788  2.53760481 -0.86784822 -0.3100495 ]]

Input = the above [5, 5] array
ReLU element wise returns the [5, 5] array
[[ 0.14431179  0.72783369  1.51149166  0.          1.78439188]
 [ 0.          0.76156765  0.          0.77131093  0.37542343]
 [ 0.49345911  0.01592223  0.38653135  0.          0.6952765 ]
 [ 0.          0.          0.          0.          0.66257358]
 [ 0.          0.34398788  2.53760481  0.          0.        ]]

Input = the above [5, 5] array
MaxPool, stride size = [1, 1], results in [4, 4] array
[[ 0.76156765  1.51149166  1.51149166  1.78439188]
 [ 0.76156765  0.76156765  0.77131093  0.77131093]
 [ 0.49345911  0.38653135  0.38653135  0.6952765 ]
 [ 0.34398788  2.53760481  2.53760481  0.66257358]]

Input = the above [4, 4] array
Fully connected layer on all 4 rows results in 5 outputs:
[ 0.08245847 -0.16351229 -0.55429065 -0.24322605 -0.99900764]
[Finished in 10.5s]
'''


