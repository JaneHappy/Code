# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/03_TensorFlow_Way_of_Linear_Regression/03_lin_reg_tensorflow_way.ipynb

from __future__ import division
from __future__ import print_function




import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets

import tensorflow as tf 
from tensorflow.python.framework import ops 

ops.reset_default_graph()
sess = tf.Session()


# Load the data
# iris.data = [(Sepal length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([ x[3]  for x in iris.data])
y_vals = np.array([ y[0]  for y in iris.data])

# Declare batch size
batch_size = 25

# Initialize placeholders
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)  #x_data *A+b = y_target
# Declare loss function (L2 loss)
loss = tf.reduce_mean(tf.square(y_target - model_output))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)
# Initialize variables
#init = tf.initialize_all_variables()
#sess.run(init)
sess.run(tf.global_variables_initializer())

# Training loop
loss_vec = []
for i in range(100):
	rand_index = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.transpose([ x_vals[rand_index] ])
	rand_y = np.transpose([ y_vals[rand_index] ])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	if (i+1)%25==0:
		print('Step #'+str(i+1) + ' \t A = '+str(sess.run(A)) + ' b = '+str(sess.run(b)) + ' \t Loss = '+str(temp_loss))

# Get the optimal coefficients
'''
x_data 		[None, 1]
y_target 	[None, 1]
A 			[1, 1]
b 			[1, 1]

model_output 	[None, 1]
	tf.matmul(x_data, A) 	[None, 1]
	tf.add(., b) 			[None, 1]+[None,1]
loss 			a number
	y_target-model_output 	[None, 1]
	tf.square(.) 			[None, 1]
	tf.reduce_mean(.)		a number 
'''
#[slope] = sess.run(A)
#[y_intercept] = sess.run(b)
[[slope      ]] = sess.run(A)
[[y_intercept]] = sess.run(b)

# Get best fit line
best_fit = []
for i in x_vals:
	best_fit.append(slope*i + y_intercept)
print(best_fit)

'''
2018-03-01 15:03:05.309642: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #25 	 A = [[ 2.06553364]] b = [[ 3.22886515]] 	 Loss = 1.02064
Step #50 	 A = [[ 1.48889542]] b = [[ 3.83931136]] 	 Loss = 0.638726
Step #75 	 A = [[ 1.24059832]] b = [[ 4.2363658]] 	 Loss = 0.357378
Step #100 	 A = [[ 1.06972659]] b = [[ 4.40965223]] 	 Loss = 0.2861
[array([ 4.62359762], dtype=float32), array([ 4.62359762], dtype=float32), array([ 4.62359762], dtype=float32), array([ 4.62359762], dtype=float32), ...., array([ 6.87002325], dtype=float32), array([ 6.44213295], dtype=float32), array([ 6.54910564], dtype=float32), array([ 6.87002325], dtype=float32), array([ 6.33516026], dtype=float32)]
[Finished in 8.1s]
'''


plt.figure(figsize=(12, 5))
plt.subplot(121)

# Plot the result
plt.plot(x_vals, y_vals,   'o',  label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width' )
plt.xlabel('Sepal Length')
#plt.show()

plt.subplot(122)

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss'   )
plt.show()





