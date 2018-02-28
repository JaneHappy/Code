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


# Regression Model
ops.reset_default_graph()
sess = tf.Session()
batch_size = 25

# Generate Data for Regression
#	Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
#	Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices  = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_trn = x_vals[train_indices]
x_vals_tst = x_vals[test_indices ]
y_vals_trn = y_vals[train_indices]
y_vals_tst = y_vals[test_indices ]

# Model Variables and Operations
#	Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
#	Add operation to graph
my_output = tf.matmul(x_data, A)  #[None,1]

# Loss, Optimization Function, and Variable Initialization
#	Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))
#	Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
#	Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Run Regression
#	Run loop
for i in range(100):
	rand_index = np.random.choice(len(x_vals_trn), size=batch_size)
	rand_x = np.transpose([x_vals_trn[rand_index]])  #[batch_size,1]
	rand_y = np.transpose([y_vals_trn[rand_index]])  #[batch_size,1]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if (i+1)%25==0:
		print('Step #'+str(i+1) + '\t A = '+str(sess.run(A)) + '\t Loss = '+str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})) )

# Evaluation of Regression Model
#	Evaluate accuracy (loss) on test set
mse_test  = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_tst]), y_target: np.transpose([y_vals_tst])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_trn]), y_target: np.transpose([y_vals_trn])})
print('MSE on test : ' + str(np.round(mse_test , 2)))
print('MSE on train: ' + str(np.round(mse_train, 2)))


'''
MSE on test : 1.11
MSE on train: 0.98
[Finished in 11.2s]
'''


# Classification Example
ops.reset_default_graph()
sess = tf.Session()
batch_size = 25

# Generate Classification Data and Targets
#	Create data
x_vals = np.concatenate((np.random.normal(-1,1,50), np.random.normal(2,1,50)))  #[100,]
y_vals = np.concatenate((np.repeat(0.,50), np.repeat(1.,50)))  #shape=(100,)
x_data   = tf.placeholder(shape=[1, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)
#	Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices  = np.array(list( set(range(len(x_vals))) - set(train_indices) ))
x_vals_trn = x_vals[train_indices]
x_vals_tst = x_vals[test_indices ]
y_vals_trn = y_vals[train_indices]
y_vals_tst = y_vals[test_indices ]

# Model Variables and Operations
#	Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
#	Add operation to graph
#	Want to create the operation sigmoid(x+A)
#	Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)

# Loss, Optimization Function, and Variable Initialization
#	Add classification loss (cross entropy)
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=my_output))
#	Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
#	Initialize variables
sess.run(tf.global_variables_initializer())  #init=

# Run Classification
#	Run loop
for i in range(1800):
	rand_index = np.random.choice(len(x_vals_trn), size=batch_size)
	#rand_x = [x_vals_trn[rand_index]]
	#rand_y = [y_vals_trn[rand_index]]
	rand_x = np.array([x_vals_trn[rand_index]])  #[1,None]
	rand_y = np.array([y_vals_trn[rand_index]])  #[1,None]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if (i+1)%200==0:
		print('Step #'+str(i+1) + '\t A = '+str(sess.run(A)) + '\t Loss = '+str( sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y}) ))

# Evaluation of Classification Results
#	Evaluate Predictions on test set
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
'''
tf.add(x_data, A)	[1,None]+[1] -> [1,None]
tf.nn.sigmoid(.)	[1,None]
tf.round(.)			[1,None]
tf.squeeze(.)		[None,]
y_target 			[1,None]
'''
correct_prediction = tf.equal(y_prediction, y_target)
#or better:  correct_prediction = tf.equal(y_prediction, tf.squeeze(y_target))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_tst = sess.run(accuracy, feed_dict={x_data: [x_vals_tst], y_target: [y_vals_tst]})
#or better:  np.array([x_vals_tst]), ..  #shape=(1,None)
acc_value_trn = sess.run(accuracy, feed_dict={x_data: [x_vals_trn], y_target: [y_vals_trn]})
print('Accuracy on train set: ', acc_value_trn)
print('Accuracy on test  set: ', acc_value_tst)

#	Plot classification result
A_result = sess.run(A)
bins = np.linspace(-5, 5, 50)

plt.figure()
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color='green') #'white')
plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red')
plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = '+str(np.round(A_result, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy = ' + str(np.round(acc_value_tst, 2)))
plt.show()


'''
2018-02-28 23:21:44.912361: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #200	 A = [[  8.64954758]], b = [[-3.52081943]]
Step #400	 A = [[ 10.24394321]], b = [[-4.63474083]]
Step #600	 A = [[ 11.11955547]], b = [[-5.43736076]]
Step #800	 A = [[ 11.80793095]], b = [[-5.94949579]]
Step #1000	 A = [[ 12.34726906]], b = [[-6.41102791]]
Step #25	 A = [[ 6.42619514]]	 Loss = 12.7847
Step #50	 A = [[ 8.63371277]]	 Loss = 2.65547
Step #75	 A = [[ 9.44932556]]	 Loss = 1.87157
Step #100	 A = [[ 9.72482777]]	 Loss = 1.56955
MSE on test : 1.44
MSE on train: 1.22
Step #200	 A = [ 3.98253369]	 Loss = 0.610938
Step #400	 A = [ 0.67315459]	 Loss = 0.399722
Step #600	 A = [-0.2335989]	 Loss = 0.298046
Step #800	 A = [-0.4328258]	 Loss = 0.184207
Step #1000	 A = [-0.46492955]	 Loss = 0.305132
Step #1200	 A = [-0.49447277]	 Loss = 0.244265
Step #1400	 A = [-0.43748763]	 Loss = 0.175463
Step #1600	 A = [-0.4472962]	 Loss = 0.266883
Step #1800	 A = [-0.4501932]	 Loss = 0.196737
Accuracy on train set:  0.9375
Accuracy on test  set:  0.95
[Finished in 39.3s]

2018-02-28 23:23:03.262407: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #200	 A = [[  8.82118797]], b = [[-3.62889647]]
Step #400	 A = [[ 10.26838875]], b = [[-4.79596472]]
Step #600	 A = [[ 11.17072582]], b = [[-5.5350275 ]]
Step #800	 A = [[ 11.92537975]], b = [[-5.95077133]]
Step #1000	 A = [[ 12.44151688]], b = [[-6.39761353]]
Step #25	 A = [[ 6.3529644]]	 Loss = 13.2608
Step #50	 A = [[ 8.65786457]]	 Loss = 1.82922
Step #75	 A = [[ 9.45079136]]	 Loss = 1.29559
Step #100	 A = [[ 9.73155022]]	 Loss = 1.24588
MSE on test : 0.75
MSE on train: 1.06
Step #200	 A = [ 6.14964437]	 Loss = 3.72025
Step #400	 A = [ 1.72587836]	 Loss = 0.354204
Step #600	 A = [-0.00845125]	 Loss = 0.29599
Step #800	 A = [-0.37620753]	 Loss = 0.335574
Step #1000	 A = [-0.51372033]	 Loss = 0.217698
Step #1200	 A = [-0.46155062]	 Loss = 0.368281
Step #1400	 A = [-0.49929941]	 Loss = 0.292356
Step #1600	 A = [-0.52270699]	 Loss = 0.268042
Step #1800	 A = [-0.51589036]	 Loss = 0.296122
Accuracy on train set:  0.925
Accuracy on test  set:  0.9
[Finished in 22.0s]
'''

## 遗留： hist 没怎么看用法



