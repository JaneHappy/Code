# coding: utf-8
# tensorflow_cookbook
#	02 TensorFlow Way
#		07 Combining Everying Together

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets
import tensorflow as tf 
from tensorflow.python.framework import ops 
ops.reset_default_graph()




#---------------------------------------
#		07 Combining Everything Together
#---------------------------------------


# Load Iris Data
#	 Load the iris data
#	 iris.target = {0, 1, 2}, where '0' is setosa
#	 iris.data ~ [sepal.width, sepal.length, pedal.width, pedal.length]
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0.  for x in iris.target])
iris_2d = np.array([[x[2], x[3]]  for x in iris.data])
#or:  iris_2d = iris.data[:, 2:]  or ..[:, 2:4]
#or:  iris_2d = iris.data[:, [2,3]]

batch_size = 20
#	Create graph
sess = tf.Session()

# Placeholders
#	Declare placeholders
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Model Variables
#	Create variables A and b
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Model Operations
#	Add model to graph: 	x1 - A*x2 + b
my_mult = tf.matmul(x2_data, A)  #[None, 1]
my_add  = tf.add(my_mult, b)     #[None, 1], every item in my_mult + b
my_output = tf.subtract(x1_data, my_add)  #[None, 1]

# Loss Function
#	Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=my_output)

# Optimizing Function and Variable Initialization
#	Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
#	Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Run Classification
#	Run loop
for i in range(1000):
	rand_index = np.random.choice(len(iris_2d), size=batch_size)
	#rand_x = np.transpose([iris_2d[rand_index]])
	rand_x = iris_2d[rand_index]
	rand_x1 = np.array([ [x[0]]  for x in rand_x])  #[batch_size,1]
	rand_x2 = np.array([ [x[1]]  for x in rand_x])  #[batch_size,1]
	#rand_y = np.transpose([binary_target[rand_index]])
	rand_y = np.array([ [y]  for y in binary_target[rand_index]])  #[batch_size,1]
	sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
	if (i+1)%200==0:
		print('Step #' + str(i+1) + '\t A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))

# Visualize Results
#	Pull out slope/intercept
[[slope]]     = sess.run(A)
[[intercept]] = sess.run(b)
#	Create fitted line
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
	ablineValues.append(slope*i+intercept)

#	Plot the fitted line over the data
setosa_x = [ a[1]  for i,a in enumerate(iris_2d)  if binary_target[i]==1]  #x2
setosa_y = [ a[0]  for i,a in enumerate(iris_2d)  if binary_target[i]==1]  #x1
non_setosa_x = [ a[1]  for i,a in enumerate(iris_2d)  if binary_target[i]==0]  #x2
non_setosa_y = [ a[0]  for i,a in enumerate(iris_2d)  if binary_target[i]==0]  #x1

plt.figure()
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'go', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width' )
plt.legend(loc='lower right')
plt.show()


'''
2018-02-28 22:18:35.575039: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #200	 A = [[  8.52632427]], b = [[-3.38528609]]
Step #400	 A = [[ 10.14555454]], b = [[-4.56772566]]
Step #600	 A = [[ 11.13013363]], b = [[-5.32883167]]
Step #800	 A = [[ 11.80720806]], b = [[-5.89874458]]
Step #1000	 A = [[ 12.33320713]], b = [[-6.37338066]]
[Finished in 38.4s]
'''

##! 这样做，是天然的“online 增量式学习器”




#---------------------------------------
#		08 Evaluating Models
#---------------------------------------











