# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/07_Implementing_Elasticnet_Regression

from __future__ import division
from __future__ import print_function



# Elastic Net Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

import tensorflow as tf
from tensorflow.python.framework import ops


## Set up for TensorFlow
ops.reset_default_graph()
#	 Create graph
sess = tf.Session()

## Obtain data
#	 Load the data
#	 iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data]) 	#[num_sample, 3]
y_vals = np.array([y[0] for y in iris.data]) 				#[num_sample,  ]


## Setup model
#	 make results reproducible
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

#	 Declare batch size
batch_size = 50

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

#	 Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b) 	#[None,1]

#	 Declare the elastic net loss function
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(tf.abs(   A)) 			#[3,1] -> a number
l2_a_loss = tf.reduce_mean(tf.square(A)) 			#[3,1] -> a number
e1_term   = tf.multiply(elastic_param1, l1_a_loss) 	#a number
e2_term   = tf.multiply(elastic_param2, l2_a_loss) 	#a number
loss      = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
'''
tf.square(y_target - model_output) 		#[None,1]
tf.reduce_mean(.) 						#a number
tf.add(., e1_term) 						#a number
tf.add(., e2_term) 						#a number
tf.expand_dims(., 0) 					#[1,]
'''

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)


## Train model
#	 Initialize variables
sess.run(tf.global_variables_initializer())

#	 Training loop
loss_vec = []
for i in range(1000):
	index  = np.random.choice(len(x_vals), size=batch_size)
	rand_x = x_vals[index] 														#[len(idx),3]
	rand_y = np.transpose([ y_vals[index] ]) 									#[len(idx),1]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) 			#
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y}) 	#[1,3,1]
	loss_vec.append(temp_loss[0]) 												#[3,1]
	if (i+1)%250==0:
		print('Step #' + str(i+1) + ' \t A = ' + str(sess.run(A)) + ' \t b = ' + str(sess.run(b)))
		print(' \t Loss = ' + str(temp_loss))

'''
2018-03-01 22:52:09.887180: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #250 	 A = [[ 1.26014626] [ 0.4016138 ] [ 0.40159121]] 	 b = [[-0.14889474]] 	 Loss = [ 1.59188581]
Step #500 	 A = [[ 1.17897248] [ 0.46715766] [ 0.29896322]] 	 b = [[-0.0677181 ]] 	 Loss = [ 1.46314824]
Step #750 	 A = [[ 1.13416564] [ 0.51899707] [ 0.21090424]] 	 b = [[ 0.01904622]] 	 Loss = [ 1.37157845]
Step #1000 	 A = [[ 1.09745109] [ 0.54604095] [ 0.13102381]] 	 b = [[ 0.10402215]] 	 Loss = [ 1.27774763]
[Finished in 4.2s]
'''


# Extract model results
#	 Get the optimal coefficients
[[sw_coef], [pl_coef], [pw_coef]] = sess.run(A)
[y_intercept] = sess.run(b)

plt.figure()
# Plot results
#	 Plot loss over time
plt.plot(loss_vec, 'k-')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.title('Loss per Generation')
plt.show()


'''
2018-03-02 01:04:21.155636: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #250 	 A = [[ 1.26014626] [ 0.4016138 ] [ 0.40159121]] 	 b = [[-0.14889474]] 	 Loss = [ 1.59188581]
Step #500 	 A = [[ 1.17897248] [ 0.46715766] [ 0.29896322]] 	 b = [[-0.0677181]] 	 Loss = [ 1.46314824]
Step #750 	 A = [[ 1.13416564] [ 0.51899707] [ 0.21090424]] 	 b = [[ 0.01904622]] 	 Loss = [ 1.37157845]
Step #1000 	 A = [[ 1.09745109] [ 0.54604095] [ 0.13102381]] 	 b = [[ 0.10402215]] 	 Loss = [ 1.27774763]
[Finished in 27.3s]
'''



