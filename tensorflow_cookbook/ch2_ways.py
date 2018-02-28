# coding: utf-8
# tensorflow_cookbook
#	02 TensorFlow Way
#		01 Operations as a Computational Graph

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import io
from tensorflow.python.framework import ops 
ops.reset_default_graph()




#---------------------------------------
#		01 Operations as a Computational Graph
# 			01 operations on a graph
#---------------------------------------
'''
# Operations on a Computational Graph
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()
# Create tensors
# Create data to feed in
x_vals = np.array([1, 3, 5, 7, 9.])
x_data = tf.placeholder(tf.float32)
m = tf.constant(3.)

# Multiplication
prod = tf.multiply(x_data, m)
for x_val in x_vals:
	print(sess.run(prod, feed_dict={x_data: x_val}))

#Output graph to Tensorboard
merged = tf.summary.merge_all(key='summaries')
if not os.path.exists('tensorboard_logs/'):
	os.makedirs('tensorboard_logs/')
my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)
'''




#---------------------------------------
#		02 Layering Nested Operations
#---------------------------------------

'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create a graph session
sess = tf.Session()
# Create the Tensors, Constants, and Placeholders

## Create data to feed in
my_array = np.array([[1.,3,5,7,9], [-2,0,2,4,6], [-6,-3,0,3,6]])
## Duplicate the array for having two inputs
x_vals = np.array([my_array, my_array+1])
## Declare the placeholder
x_data = tf.placeholder(tf.float32, shape=(3,5))
## Declare constants for operations
m1 = tf.constant([[1.], [0], [-1], [2], [4]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# Declare Operations
## 1st Operation Layer = Multiplication
prod1 = tf.matmul(x_data, m1)
## 2nd Operation Layer = Multiplication
prod2 = tf.matmul(prod1, m2)
## 3rd Operation Layer = Addition
add1 = tf.add(prod2, a1)

# Evaluate and Print Output
for x_val in x_vals:
	print(sess.run(add1, feed_dict={x_data: x_val}))

#----- my
for x_val in x_vals:
	print("x_data * m1")
	print("\t", sess.run( tf.matmul(  x_data, m1) , feed_dict={x_data: x_val} ))
	#error! print("\t", sess.run( tf.multiply(x_data, m1) , feed_dict={x_data: x_val} ))  #error!
	print("prod1 * m2")
	print("\t", sess.run( tf.matmul(tf.matmul(x_data, m1), m2) , feed_dict={x_data: x_val} ))
	#error! print("\t", sess.run( tf.matmul(tf.multiply(x_data, m1), m2) , feed_dict={x_data: x_val} ))
	#error! print("\t", sess.run( tf.multiply(tf.matmul(x_data, m1), m2) , feed_dict={x_data: x_val} ))
	#error! print("\t", sess.run( tf.multiply(tf.multiply(x_data, m1), m2) , feed_dict={x_data: x_val} ))
	print("prod2 + a1")
	print("\t", sess.run( tf.add(tf.matmul(tf.matmul(x_data, m1), m2), a1) , feed_dict={x_data: x_val} ))


# Create and Format Tensorboard outputs for viewing
merged = tf.summary.merge_all(key='summaries')
if not os.path.exists('tensorboard_logs/'):
	os.makedirs('tensorboard_logs/')
my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)
'''

'''
2018-02-28 15:19:56.674010: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[[ 102.]
 [  66.]
 [  58.]]
[[ 114.]
 [  78.]
 [  70.]]
[Finished in 1.8s]
'''




#---------------------------------------
#		03 Working with Multiple Layers
#---------------------------------------

'''

sess = tf.Session()
## Create a small random 'image' of size 4x4
x_shape = [1, 4, 4, 1]
x_val  = np.random.uniform(size=x_shape)
x_data = tf.placeholder(tf.float32, shape=x_shape)

# First Layer: Moving Window (Convolution)
#	 Create a layer that takes a spatial moving window average
#	 Our window will be 2x2 with a stride of 2 for height and width
#	 The filter value will be 0.25 because we want the average of the 2x2 window
my_filter  = tf.constant(0.25, shape=[2,2,1,1])
my_strides = [1,2,2,1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')

# Second Layer: Custom
#	 Define a custom layer which will be sigmoid(Ax+b) where
#	 x is a 2x2 matrix and A and b are 2x2 matrices
def custom_layer(input_matrix):
	input_matrix_sqeezed = tf.squeeze(input_matrix)
	A = tf.constant([[1., 2], [-1, 3]])
	b = tf.constant(1., shape=[2, 2])
	temp1 = tf.matmul(A, input_matrix_sqeezed)
	temp = tf.add(temp1, b)  #Ax + b
	return(tf.sigmoid(temp))
#	 Add custom layer to graph
with tf.name_scope('Custom_Layer') as scope:
	custom_layer1 = custom_layer(mov_avg_layer)

# Run Output
print(sess.run(mov_avg_layer, feed_dict={x_data: x_val}))
print(sess.run(custom_layer1, feed_dict={x_data: x_val}))

'''


'''
2018-02-28 16:01:46.032314: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[[[[ 0.56041563]
   [ 0.61760235]]
  [[ 0.17081998]
   [ 0.43227127]]]]
[Finished in 2.0s]

2018-02-28 16:02:29.205503: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[[[[ 0.67536056]
   [ 0.69821018]]
  [[ 0.36045432]
   [ 0.66382778]]]]
[[ 0.91654235  0.95372909]
 [ 0.80313462  0.90831834]]
[Finished in 1.7s]
'''




#---------------------------------------
#		04 Implementing Loss Functions
#---------------------------------------




#---------------------------------------
#		05 Implementing Back Propagation
#---------------------------------------




#---------------------------------------
#		06 Working with Batch and Stochastic Training
#---------------------------------------




#---------------------------------------
#		07 Combining Everything Together
#---------------------------------------




#---------------------------------------
#		08 Evaluating Models
#---------------------------------------





