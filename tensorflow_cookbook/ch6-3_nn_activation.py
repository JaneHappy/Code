# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/03_Working_with_Activation_Functions

from __future__ import division
from __future__ import print_function


# Combining Gates and Activation Functions
import tensorflow as tf 
from tensorflow.python.framework import ops 
ops.reset_default_graph()

import numpy as np 
import matplotlib.pyplot as plt 


#	 Start Graph Session
sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

batch_size = 50 

a1 = tf.Variable(tf.random_normal( shape=[1, 1]))
b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
a2 = tf.Variable(tf.random_normal( shape=[1, 1]))
b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))
x  = np.random.normal(2, 0.1, 500)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1)) 			#[None,1]

relu_activation    = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2)) 			#[None,1]


#	 Declare the loss function as the difference between
#	 the output and a target value, 0.75.
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75))) 	#a number
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation   , 0.75))) 	#a number

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)  #can be same!
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu    = my_opt.minimize(loss2)

#	 Run loop across gate
print('\nOptimizing Sigmoid AND Relu Output to 0.75')
loss_vec_sigmoid = []
loss_vec_relu    = []
for i in range(500):
	indices = np.random.choice(len(x), size=batch_size)
	x_vals  = np.transpose([ x[indices] ])  #[batch_size,1]
	sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
	sess.run(train_step_relu   , feed_dict={x_data: x_vals})

	loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals})) 	#a number
	loss_vec_relu.append(   sess.run(loss2, feed_dict={x_data: x_vals})) 	#a number

	#sigmoid_output = np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals})) 	#[None,1] -> a number
	#relu_output    = np.mean(sess.run(relu_activation   , feed_dict={x_data: x_vals})) 	#[None,1] -> a number
	sigmoid_output = sess.run(sigmoid_activation, feed_dict={x_data: x_vals})
	relu_output    = sess.run(relu_activation   , feed_dict={x_data: x_vals})

	if i%50==0:
		print('sigmoid = '+str(np.mean(sigmoid_output)) + ' relu = '+str(np.mean(relu_output)))

'''
2018-03-02 10:34:38.888669: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX

Optimizing Sigmoid AND Relu Output to 0.75
sigmoid = 0.126552 relu = 2.02276
sigmoid = 0.178638 relu = 0.75303
sigmoid = 0.247698 relu = 0.74929
sigmoid = 0.344675 relu = 0.749955
sigmoid = 0.440066 relu = 0.754
sigmoid = 0.52369  relu = 0.754772
sigmoid = 0.583739 relu = 0.75087
sigmoid = 0.627335 relu = 0.747023
sigmoid = 0.65495  relu = 0.751805
sigmoid = 0.674526 relu = 0.754707
[Finished in 13.8s]
'''

#	 Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu   , 'r--', label='Relu Activation')
#plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


