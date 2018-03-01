# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.ipynb

from __future__ import division
from __future__ import print_function

#	 import required libraries
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

import tensorflow as tf
from tensorflow.python.framework import ops


#	 Specify 'Ridge' or 'LASSO'
regression_type = 'LASSO'
#regression_type = 'Ridge'

#	 clear out old graph
ops.reset_default_graph()

#	 Create graph
sess = tf.Session()


# Load iris data
#	 iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Model Parameters
#	 Declare batch size
batch_size = 50

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 make results reproducible
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

#	 Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

#	 Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)  #[None, 1]


# Loss Functions
#	 Select appropriate loss function based on regression type

if regression_type == 'LASSO':
	#	 Declare Lasso loss function
	#	 Lasso Loss = L2_Loss + heavyside_step,
	#	 Where heavyside_step ~ 0 if A < constant, otherwise ~ 99
	lasso_param = tf.constant(0.9)
	heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-50., tf.subtract(A, lasso_param)))))
	regularization_param = tf.multiply(heavyside_step, 99.)
	loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)

elif regression_type == 'Ridge':
	#	 Declare the Ridge loss function
	#	 Ridge loss = L2_loss + L2 norm of slope
	ridge_param = tf.constant(1.)
	ridge_loss  = tf.reduce_mean(tf.square(A))
	loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)

else:
	print('Invalid regression_type parameter value', file=sys.stderr)

''' 
heavyside_step = 1/(1+e^(-50*(A-0.9)) 					#[1,1]
loss = mean(( y-xA-b )^2) + heavyside_step*99 			#a number + [1,1] -> [1,1]

rigde_loss = mean(A^2)
	rigde_step_a = tf.reduce_mean(tf.square(y_target - model_output))
	ridge_step_b = tf.multiply(ridge_param, ridge_loss)
	ridge_step_c = tf.add( ridge_step_a , ridge_step_b )
ridge_step_a = mean(( y-xA-b )^2) 						#[None,1] -> a number
ridge_step_b = 1.*mean(A^2) 							#[1,1]
ridge_step_c = mean(( y-xA-b )^2) + 1.*mean(A^2) 		#[1,1]
loss =  												#[1,1,1]
'''


# Optimizer
#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Run regression
#	 Initialize variables
sess.run(tf.global_variables_initializer())  #init

#	 Training loop
loss_vec = []
for i in range(1500):
	index = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.transpose([ x_vals[index] ])
	rand_y = np.transpose([ y_vals[index] ])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss[0])
	if (i+1)%300==0:
		print('Step #' + str(i+1) + ' \t A = ' + str(sess.run(A)) + ' \t b = ' + str(sess.run(b)) + ' \t Loss = ' + str(temp_loss))
		#print('\n')


'''
LASSO
2018-03-01 20:13:34.543854: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #300 	 A = [[ 0.77170753]] 	 b = [[ 1.82499862]] 	 Loss = [[ 10.26473045]]
Step #600 	 A = [[ 0.75908542]] 	 b = [[ 3.2220633 ]] 	 Loss = [[  3.06292033]]
Step #900 	 A = [[ 0.74843585]] 	 b = [[ 3.9975822 ]] 	 Loss = [[  1.23220456]]
Step #1200 	 A = [[ 0.73752165]] 	 b = [[ 4.42974091]] 	 Loss = [[  0.57872057]]
Step #1500 	 A = [[ 0.72942668]] 	 b = [[ 4.67253113]] 	 Loss = [[  0.40874988]]
[Finished in 15.5s]

Ridge
2018-03-01 20:15:32.777660: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #300 	 A = [[ 1.70595658]] 	 b = [[ 1.55541849]] 	 Loss = [ 8.26485252]
Step #600 	 A = [[ 1.61488783]] 	 b = [[ 2.56744456]] 	 Loss = [ 4.79188538]
Step #900 	 A = [[ 1.34542716]] 	 b = [[ 3.24855161]] 	 Loss = [ 3.07603455]
Step #1200 	 A = [[ 1.11086905]] 	 b = [[ 3.76259446]] 	 Loss = [ 2.06162453]
Step #1500 	 A = [[ 0.93269861]] 	 b = [[ 4.15556765]] 	 Loss = [ 1.48733997]
[Finished in 4.3s]
'''


# Extract regression results
#	 Get the optimal coefficients
[[slope    ]] = sess.run(A)
[[intercept]] = sess.run(b)

#	 Get the optimal coefficients
best_fit = []
for i in x_vals:
	best_fit.append( slope*i + intercept )


# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
#	 %matplotlib inline
#	 Plot the result
plt.plot(x_vals, y_vals,   'o',  label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width' )
plt.ylabel('Sepal Length')
#plt.show()

plt.subplot(122)
#	 Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title(regression_type + ' Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()


'''
exe3: Ridge
2018-03-01 20:20:50.995184: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #300 	 A = [[ 1.70595658]] 	 b = [[ 1.55541849]] 	 Loss = [ 8.26485252]
Step #600 	 A = [[ 1.61488783]] 	 b = [[ 2.56744456]] 	 Loss = [ 4.79188538]
Step #900 	 A = [[ 1.34542716]] 	 b = [[ 3.24855161]] 	 Loss = [ 3.07603455]
Step #1200 	 A = [[ 1.11086905]] 	 b = [[ 3.76259446]] 	 Loss = [ 2.06162453]
Step #1500 	 A = [[ 0.93269861]] 	 b = [[ 4.15556765]] 	 Loss = [ 1.48733997]
[Finished in 28.3s]

exe4: Lasso
2018-03-01 20:21:41.317912: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #300 	 A = [[ 0.77170753]] 	 b = [[ 1.82499862]] 	 Loss = [[ 10.26473045]]
Step #600 	 A = [[ 0.75908542]] 	 b = [[ 3.2220633]] 	 Loss = [[ 3.06292033]]
Step #900 	 A = [[ 0.74843585]] 	 b = [[ 3.9975822]] 	 Loss = [[ 1.23220456]]
Step #1200 	 A = [[ 0.73752165]] 	 b = [[ 4.42974091]] 	 Loss = [[ 0.57872057]]
Step #1500 	 A = [[ 0.72942668]] 	 b = [[ 4.67253113]] 	 Loss = [[ 0.40874988]]
[Finished in 24.2s]
'''


