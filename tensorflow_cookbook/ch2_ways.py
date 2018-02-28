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
#	https://github.com/nfmcclure/tensorflow_cookbook/blob/master/02_TensorFlow_Way/04_Implementing_Loss_Functions/04_loss_functions.ipynb
#---------------------------------------

sess = tf.Session()


'''
# Numerical Predictions

#	 Various Predicted X-values
x_vals = tf.linspace(-1., 1, 500)
#	 Create our target of zero
target = tf.constant(0.)

## L2 Loss
#	L2 loss:	L = (pred - actual)^2
l2_y_vals = tf.square(target - x_vals)
l2_y_out  = sess.run(l2_y_vals)

## L1 Loss
#	L1 loss:	L = abs(pred - actual)
l1_y_vals = tf.abs(target - x_vals)
l1_y_out  = sess.run(l1_y_vals)

## Pseudo-Huber Loss
#	The psuedo-huber loss function is a smooth approximation to the L1 loss as the (predicted - target) values get larger. When the predicted values are close to the target, the pseudo-huber loss behaves similar to the L2 loss.
# 	L = delta^2 * ( sqrt( 1 + ((pred - actual) / delta)^2 ) -1)
#	Pseudo-Huber with delta = 0.25
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out  = sess.run(phuber1_y_vals)
#	Pseudo-Huber with delta = 5
delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out  = sess.run(phuber2_y_vals)

## Plot the Regression Losses
x_array = sess.run(x_vals)
plt.figure()
plt.plot(x_array, l2_y_out, 'b-' , label='L2 Loss' )
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss' )
plt.plot(x_array, phuber1_y_out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:' , label='P-Huber Loss (5.0 )')
plt.ylim(-0.2, 0.4) #annotation
plt.legend(loc='lower right', prop={'size': 11})
plt.show()


2018-02-28 17:01:20.020701: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 95.1s]
'''


'''

# Categorical Predictions
#	Various predicted X values
x_vals = tf.linspace(-3., 5, 500)
#	Target of 1.0
target = tf.constant(1.)
targets = tf.fill([500,], 1.)  #or tf.fill([500], 1.)

## Hinge Loss
#	Use for predicting binary (-1, 1) classes
#	L = max(0, 1 - (pred * actual))
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out  = sess.run(hinge_y_vals)

## Cross Entropy Loss
#	L = -actual * (log(pred)) - (1-actual)(log(1-pred))
xentropy_y_vals = -tf.multiply(target, tf.log(x_vals)) - tf.multiply((1.-target), tf.log(1.-x_vals))  #tf.log=ln()
xentropy_y_out  = sess.run(xentropy_y_vals)

## Sigmoid Entropy Loss
#	L = -actual * log(sigmoid(pred)) - (1-actual) * log(1-sigmoid(pred))
#	or
#	L = max(actual, 0) - actual * pred + log(1. + exp(-abs(actual)))
x_val_input  = tf.expand_dims(x_vals , 1)  #axis=1
target_input = tf.expand_dims(targets, 1)
xentropy_sigmoid_y_vals = tf.nn.softmax_cross_entropy_with_logits(logits=x_val_input, labels=target_input)
xentropy_sigmoid_y_out  = sess.run(xentropy_sigmoid_y_vals)

my_out_1 = sess.run( -tf.multiply(targets, tf.log(tf.sigmoid(x_vals))) -tf.multiply(1.-targets, tf.log(1.-tf.sigmoid(x_vals))) )  #tf.subtract()
my_out_2 = sess.run( tf.maximum(targets, 0.) - tf.multiply(targets, x_vals) + tf.log(1.+tf.exp(-tf.abs(targets))) )

## Weighted (Softmax) Cross Entropy Loss
#	L = -actual * log(pred) * weights - (1-actual) * log(1-pred)
#	or
#	L = (1-pred)*actual + (1+ (weights-1)*pred) * log(1+exp(-actual))
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
xentropy_weighted_y_out  = sess.run(xentropy_weighted_y_vals)

my_weight = tf.fill([500], 0.5)
my_out_3 = sess.run( -tf.multiply(tf.multiply(targets, tf.log(x_vals)), my_weight) -tf.multiply(1.-targets, tf.log(1.-x_vals)) )
my_out_4 = sess.run( tf.multiply(1.-x_vals, targets) + tf.multiply(1.+tf.multiply(my_weight-1., x_vals), tf.log(1.+tf.exp(-targets))) )

## Plot the Categorical Losses
#	Plot the output
x_array = sess.run(x_vals)
plt.figure()
plt.plot(x_array, hinge_y_out   , 'b-' , label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out , 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_out, 'g:' , label='Weighted Cross Entropy Loss (x0.5)')
#plt.ylim(-1.5, 3)
#plt.xlim(-1, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(x_array, xentropy_sigmoid_y_out, 'r-', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, my_out_1, 'b--', label='First  Calculation')
plt.plot(x_array, my_out_2, 'g:' , label='Second Calculation')
plt.legend(loc='lower right')
plt.subplot(122)
plt.plot(x_array, xentropy_weighted_y_out, 'r-', label='Weighted Cross Entropy Loss')
plt.plot(x_array, my_out_3, 'b--', label='First  Calculation')
plt.plot(x_array, my_out_4, 'g:' , label='Second Calculation')
plt.legend(loc='lower right')
plt.show()


2018-02-28 17:40:47.500771: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 110.9s]
'''


#' ''
# Softmax entropy and Sparse Entropy
#	Softmax entropy loss
#	L = -actual * log(softmax(pred)) - (1-actual) * log(1-softmax(pred))
unscaled_logits = tf.constant([[1., -3, 10]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist)
print(sess.run(softmax_xentropy))
# 	Sparse entropy loss
#	Use when classes and targets have to be mutually exclusive
#	L = sum( -actual * log(pred) )
unscaled_logits = tf.constant([[1., -3, 10]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=sparse_target_dist)
print(sess.run(sparse_xentropy))

'''
2018-02-28 17:49:16.563392: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[ 1.16012561]
[ 0.00012564]
[Finished in 1.5s]
'''




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





