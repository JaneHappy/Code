# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/05_Implementing_Deming_Regression

from __future__ import division
from __future__ import print_function




# Implementing Deming Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()



#	 Load the data
#	 iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris   = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data]) # Petal Width
y_vals = np.array([y[0] for y in iris.data]) # Sepal Length

#	 Declare batch size
batch_size = 125

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

#	 Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b) 	#[None,1]

#	 Declare Demming loss function
demming_numerator   = tf.abs(tf.subtract(tf.add(tf.matmul(x_data, A), b), y_target))  # a number (x)
#					tf.matmul(x_data, A) 		[None,1]
#					tf.add(., b) 				[None,1]
#					tf.subtract(., y_target)	[None,1]
#					tf.abs(.) 					[None,1]
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1)) 	#[1,1]
loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.25)
train_step = my_opt.minimize(loss)

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Training loop
loss_vec = []
for i in range(1500):
	index  = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.transpose([ x_vals[index] ])
	rand_y = np.transpose([ y_vals[index] ])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	if (i+1)%100==0:
		print('Step #'+str(i+1) + ' \t A = '+str(sess.run(A)) + ' \t b = '+str(sess.run(b)) + ' \t Loss = '+str(temp_loss))



