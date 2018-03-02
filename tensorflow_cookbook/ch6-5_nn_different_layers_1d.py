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


#	 Create graph session 
ops.reset_default_graph()
sess = tf.Session()

#	 parameters for the run
data_size    = 25
conv_size    = 5
maxpool_size = 5
stride_size  = 1#2

#	 ensure reproducibility
seed=13
np.random.seed(seed)
tf.set_random_seed(seed)

#	 Generate 1D data
data_1d = np.random.normal(size=data_size)

#	 Placeholder
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])



#--------Convolution--------
def conv_layer_1d(input_1d, my_filter, stride):
	#	 TensorFlow's 'conv2d()' function only works with 4D arrays:
	#	 [batch#, width, height, channels], we have 1 batch, and
	#	 width = 1, but height = the length of the input, and 1 channel.
	#	 So next we create the 4D array by inserting dimension 1's.
	input_2d = tf.expand_dims(input_1d, 0)  #[1, data_size]
	input_3d = tf.expand_dims(input_2d, 0)  #[1,1, data_size]
	input_4d = tf.expand_dims(input_3d, 3)  #[1,1, data_size, 1]
	#	 Perform convolution with stride = 1, if we wanted to increase the stride,
	#	 to say '2', then strides=[1,1,2,1]
	convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,stride,1], padding='VALID')
	#	 Get rid of extra dimensions
	conv_output_1d = tf.squeeze(convolution_output)
	return(conv_output_1d)

#	 Create filter for convolution.
my_filter = tf.Variable(tf.random_normal(shape=[1,conv_size,1,1]))
#	 Create convolution layer
my_convolution_output = conv_layer_1d(x_input_1d, my_filter, stride=stride_size)

'''
x_input_1d 		#[data_size,]
	input_4d 	#[1,1, data_size, 1]
my_filter 		#[1, conv_size, 1,1]
	(1) filter   			-> 	[1*conv_size*1, 1]
	(2) input_4d 			-> 	[1, out_h,out_w, 1*conv_size*1]
	(3) input_4d' * filter' -> 	[1, out_h,out_w, 1]
strides 		#[1,1, stride_size, 1]
	(i ) out_h = ceil( (1-1+1)/1 )							   =  ceil(1) = 1
	(ii) out_w = ceil( (data_size-conv_size+1)/stride_size ) =sp.  ceil((25-5+1)/1)=21  or ceil((25-5+1)/2)=ceil(10.5)=11

therefore: 
	convolution_output 		#[1, out_h,out_w, 1]    i.e., [1,1,21,1] or [1,1,11,1]
	conv_output_1d 			#[out_h, out_w] 		i.e., [21,]  or [11,] 				i.e., [out_w,]
	my_convolution_output 	#[out_w,]   			i.e., ceil( (data_size-conv_size+1)/stride_size )
'''



#--------Activation--------
def activation(input_1d):
	return(tf.nn.relu(input_1d))

#	 Create activation layer
my_activation_output = activation(my_convolution_output)

'''
input_1d 		#[out_w,]
tf.nn.relu(.) 	#[out_w,]
	my_activation_output 	#[out_w,]
'''



#--------Max Pool--------
def max_pool(input_1d, width, stride):
	#	 Just like 'conv2d()' above, max_pool() works with 4D arrays.
	#	 [batch_size=1, width=1, height=num_input, channels=1]
	input_2d = tf.expand_dims(input_1d, 0) 		#[1, out_w]
	input_3d = tf.expand_dims(input_2d, 0) 		#[1,1, out_w]
	input_4d = tf.expand_dims(input_3d, 3) 		#[1,1, out_w, 1]
	#	 Perform the max pooling with strides = [1,1,1,1]
	#	 If we wanted to increase the stride on our data dimension, say by
	#	 a factor of '2', we put strides = [1, 1, 2, 1]
	#	 We will also need to specify the width of the max-window ('width')
	pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
								 strides=[1, 1, stride, 1], padding='VALID')
	#	 Get rid of extra dimensions
	pool_output_1d = tf.squeeze(pool_output)
	return(pool_output_1d)

my_maxpool_output = max_pool(my_activation_output, width=maxpool_size, stride=stride_size)

'''
input_1d 		#[out_w,]
	input_4d 	#[1,1, out_w, 1]
	k_size 		#[1,1, maxpool_size, 1]
	strides 	#[1,1, stride_size,  1]

	?	pool_output 	#[1,  (1-1)/1+1, (out_w-maxpool_size)/stride_size+1, 1 ]
						#sp,  [1, 1,(out_w-5)/stride_size+1,1]
						#	if stride_size=1 	out_w=21, [1,1,16/1+1,1], [1,1,17,1]
						#	if stride_size=2 	out_w=11, [1,1, 6/2+1,1], [1,1, 4,1]
		pool_output_1d 	[(out_w - maxpool_size)/stride_size+1 ,] 		i.e., [17,] or [4,]
my_maxpool_output		#[ (out_w - maxpool_size)/stride_size +1 ,]
'''



#--------Fully Connected--------
def fully_connected(input_layer, num_outputs):
	#	 First we find the needed shape of the multiplication weight matrix:
	#	 The dimension will be (length of input) by (num_outputs)
	weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs] ]))
	'''
	tf.shape(input_layer) 			#[ (out_w-maxpool_size)/stride_size+1 ,]
	tf.stack([., [num_outputs]]) 	#output: [[(out_w-maxpool_size)/stride_size+1], [num_outputs]], tf.int32
	tf.squeeze(.) 					#output: [(out_w-maxpool_size)/stride_size+1, num_outputs] 		tf.int32
	'''
	#	 Initialize such weight
	weight = tf.random_normal(weight_shape, stddev=0.1) 	#[(out_w-maxpool_size)/stride_size+1, num_outputs]
	#	 Initialize the bias
	bias   = tf.random_normal(shape=[num_outputs]) 			#[num_outputs,]
	#	 Make the 1D input array into a 2D array for matrix multiplication
	input_layer_2d = tf.expand_dims(input_layer, 0) 		#[1, (out_w-maxpool_size)/stride_size+1 ]
	#	 Perform the matrix multiplication and add the bias
	full_output = tf.add(tf.matmul(input_layer_2d, weight), bias) 	#[1,num_outputs]
	#	 Get rid of extra dimensions
	full_output_1d = tf.squeeze(full_output) 						#[num_outputs,]
	return(full_output_1d)

my_full_output = fully_connected(my_maxpool_output, 5)

'''
my_maxpool_output			#[ (out_w - maxpool_size)/stride_size +1 ,]
full_output_1d 				#[num_outputs,]

my_full_output 				#[num_outputs,] 	i.e., [5,]
'''



# Run graph
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}
'''
data_size = number of samples, not number of features

updated: Nono, this is a sample, which has 'data_size' features.
updated: Previously, I thought it was '3' after pooling when 'stride_size=2'. It was wrong.
'''

print('>>>> 1D Data <<<<')

#	 Convolution Output
print('Input = array of length         %d' % ( x_input_1d.shape.as_list()[0] ))
print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:' % (conv_size, stride_size, my_convolution_output.shape.as_list()[0] ))
print(sess.run(my_convolution_output, feed_dict=feed_dict))

#	 Activation Output
print('\nInput = above array of length %d' % ( my_convolution_output.shape.as_list()[0] ))
print('ReLU element wise returns an array of length %d:' % ( my_activation_output.shape.as_list()[0] ))
print(sess.run(my_activation_output, feed_dict=feed_dict))

#	 Max Pool Output
print('\nInput = above array of length %d' % (my_activation_output.shape.as_list()[0]))
print('MaxPool, window length = %d, stride size = %d, results in the array of length %d' % (maxpool_size,stride_size,my_maxpool_output.shape.as_list()[0] ))
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

#	 Fully Connected Output
print('\nInput = above array of length %d' % (my_maxpool_output.shape.as_list()[0]))
print('Fully connected layer on all 4 rows with %d outputs:' % (my_full_output.shape.as_list()[0]))
print(sess.run(my_full_output, feed_dict=feed_dict))


'''
exe-1th: stride_size=1
2018-03-02 12:30:46.552190: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
>>>> 1D Data <<<<
Input = array of length         25
Convolution w/ filter, length = 5, stride size = 1, results in an array of length 21:
[-2.63576341 -1.11550486 -0.95571411 -1.69670296 -0.35699379  0.62266493
  4.43316031  2.01364899  1.33044648 -2.30629659 -0.82916248 -2.63594174
  0.76669347 -2.46465087 -2.2855041   1.49780679  1.6960566   1.48557389
 -2.79799461  1.18149185  1.42146575]
Input = above array of length 21
ReLU element wise returns an array of length 21:
[ 0.          0.          0.          0.          0.          0.62266493
  4.43316031  2.01364899  1.33044648  0.          0.          0.
  0.76669347  0.          0.          1.49780679  1.6960566   1.48557389
  0.          1.18149185  1.42146575]
Input = above array of length 21
MaxPool, window length = 5, stride size = 1, results in the array of length 17
[ 0.          0.62266493  4.43316031  4.43316031  4.43316031  4.43316031
  4.43316031  2.01364899  1.33044648  0.76669347  0.76669347  1.49780679
  1.6960566   1.6960566   1.6960566   1.6960566   1.6960566 ]
Input = above array of length 17
Fully connected layer on all 4 rows with 5 outputs:
[ 1.71536088 -0.72340977 -1.22485089 -2.5412786  -0.16338299]
[Finished in 11.0s]

exe-2nd: stride_size=2
2018-03-02 12:32:22.458753: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
>>>> 1D Data <<<<
Input = array of length         25
Convolution w/ filter, length = 5, stride size = 2, results in an array of length 11:
[-2.63576341 -0.95571411 -0.35699379  4.43316031  1.33044648 -0.82916248
  0.76669347 -2.2855041   1.6960566  -2.79799461  1.42146575]
Input = above array of length 11
ReLU element wise returns an array of length 11:
[ 0.          0.          0.          4.43316031  1.33044648  0.
  0.76669347  0.          1.6960566   0.          1.42146575]
Input = above array of length 11
MaxPool, window length = 5, stride size = 2, results in the array of length 4
[ 4.43316031  4.43316031  1.6960566   1.6960566 ]
Input = above array of length 4
Fully connected layer on all 4 rows with 5 outputs:
[ 1.1263864  -0.30834734  0.28539652 -3.01429224  0.47790092]
[Finished in 6.7s]
'''





