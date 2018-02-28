# coding: utf-8
# tensorflow_cookbook
#	02 TensorFlow Way
#		01 Operations as a Computational Graph
#		...
#		06 Working with Batch and Stochastic Training

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
'''

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


'''
2018-02-28 17:01:20.020701: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 95.1s]
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
plt.ylim(-1.5, 3)
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


'''
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



sess = tf.Session()

# A Regression Example
x_vals = np.random.normal(1, 0.1, 100)  # a normal (mean of 1.0, stdev of 0.1)
y_vals = np.repeat(10., 100)
x_data   = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
#	Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))
#	Add operation to graph
my_output = tf.multiply(x_data, A)
#	Add L2 loss operation to graph
loss = tf.square(my_output - y_target)
#	Initialize variables
init = tf.global_variables_initializer()
sess.run(init)
#	Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)  #learning_rate
train_step = my_opt.minimize(loss)

# Running the Regression Graph
#	Run Loop
for i in range(100):
	rand_index = np.random.choice(100)  #.choice(maximum, size=(..))
	rand_x = [x_vals[rand_index]]
	rand_y = [y_vals[rand_index]]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if (i+1)%25==0:
		print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
		print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))


# Classification Example
#	Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))  #size=(100,)
x_data   = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
#	Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

#	 Add operation to graph
#	 Want to create the operstion sigmoid(x + A)
#	 Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)
#	 Now we have to add another dimension to each (batch size of 1)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded  = tf.expand_dims(y_target , 0)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)
#	 Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Running the Classification Graph
#	Run loop
for i in range(1400):
	rand_index = np.random.choice(100)
	rand_x = [x_vals[rand_index]]
	rand_y = [y_vals[rand_index]]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if (i+1)%200==0:
		print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
		print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))


# 	Evaluate Predictions
predictions = []
for i in range(len(x_vals)):
	x_val = [x_vals[i]]
	prediction = sess.run(tf.round(tf.sigmoid(my_output)), feed_dict={x_data: x_val})
	predictions.append(prediction[0])
accuracy = sum(x==y for x,y in zip(predictions, y_vals))/100.
print('Ending Accuracy = ' + str(np.round(accuracy, 2)))



'''
2018-02-28 18:29:40.234289: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #25 	A = [ 6.11608648]	Loss = [ 16.08420181]
Step #50 	A = [ 8.52250576]	Loss = [  5.10827351]
Step #75 	A = [ 9.24069977]	Loss = [  1.14894378]
Step #100 	A = [ 9.71889877]	Loss = [  0.4407751 ]
[Finished in 2.0s]

2018-02-28 18:44:25.064733: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[ 1.16012561]
[ 0.00012564]
Step #25 	A = [ 6.87281036]	Loss = [ 14.14905739]
Step #50 	A = [ 8.9189167 ]	Loss = [  1.28802204]
Step #75 	A = [ 9.77301693]	Loss = [  0.3833788 ]
Step #100 	A = [ 9.62869167]	Loss = [  2.50389123]
Step #200 		A = [ 6.78434467]	Loss = [[  3.24778448e-05]]
Step #400 		A = [ 2.04075623]	Loss = [[ 1.83606267]]
Step #600 		A = [-0.15876967]	Loss = [[ 0.53752458]]
Step #800 		A = [-0.83228946]	Loss = [[ 0.01623671]]
Step #1000 		A = [-0.96144277]	Loss = [[ 0.03637504]]
Step #1200 		A = [-1.08885336]	Loss = [[ 0.30568787]]
Step #1400 		A = [-0.82672775]	Loss = [[ 0.1000525 ]]
[Finished in 3.0s]

2018-02-28 18:50:33.832883: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[ 1.16012561]
[ 0.00012564]
Step #25 	A = [ 6.13512325]	Loss = [ 7.87540627]
Step #50 	A = [ 8.50735569]	Loss = [ 0.14130819]
Step #75 	A = [ 9.42903709]	Loss = [  7.48194871e-05]
Step #100 	A = [ 9.65153313]	Loss = [ 1.11493409]
Step #200 		A = [ 4.85303402]	Loss = [[ 4.48928118]]
Step #400 		A = [ 1.11574042]	Loss = [[ 0.0349636 ]]
Step #600 		A = [-0.5154478 ]	Loss = [[ 0.02781134]]
Step #800 		A = [-0.96965766]	Loss = [[ 0.08468418]]
Step #1000 		A = [-1.10859692]	Loss = [[ 1.09603882]]
Step #1200 		A = [-1.22808003]	Loss = [[ 0.2578792 ]]
Step #1400 		A = [-1.28782296]	Loss = [[ 0.11318245]]
Ending Accuracy = 0.96
[Finished in 4.3s]
'''




#---------------------------------------
#		06 Working with Batch and Stochastic Training
#---------------------------------------


# Stochastic Training
print("Chapter 2.6 \t Stochastic Training")
ops.reset_default_graph()
sess = tf.Session()

## Generate Data
x_vals = np.random.normal(1, 0.1, 100)  #a Normal (mean=1, sd=0.1)
y_vals = np.repeat(10., 100)
x_data   = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

## Model Variables and Operations
#	Create variables (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))
#	Add operation to graph
my_output = tf.multiply(x_data, A)

## Loss Function
#	Add L2 loss operation to graph
loss = tf.square(my_output - y_target)

## Optimization and Initialization
#	Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
#	Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

## Train Model
loss_stochastic = []
#	Run loop
for i in range(100):
	rand_index = np.random.choice(100)
	rand_x = [x_vals[rand_index]]
	rand_y = [y_vals[rand_index]]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if (i+1)%5==0:
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		print('Step #' + str(i+1) + '\tA = ' + str(sess.run(A)) + '\tLoss = ' + str(temp_loss))
		loss_stochastic.append(temp_loss)

accuracy_stochastic = []
for x_val in x_vals:
	accuracy_stochastic.append(sess.run( my_output, feed_dict={x_data: [x_val]} ))
#accuracy_stochastic = sum(x==y for x,y in zip(accuracy_stochastic, y_vals))/100.
#accuracy_stochastic = np.mean([x==y  for x,y in zip(accuracy_stochastic, y_vals)])
accuracy_stochastic = np.mean([ np.abs(x-y)<=1e-6  for x,y in zip(accuracy_stochastic, y_vals)])


# Batch Training
print("Chapter 2.6 \t Batch Training")
#	Re-initialize graph
ops.reset_default_graph()
sess = tf.Session()
#	Declare batch size
batch_size = 25

## Generate Data
x_vals = np.random.normal(1, 0.1, 100)  #a Normal (mean=1, sd=0.1)
y_vals = np.repeat(10., 100)
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

## Model Variables and Operations
#	Create variables (one model parameter = B)
B = tf.Variable(tf.random_normal(shape=[1,1]))
#	Add operation to graph
my_output = tf.matmul(x_data, B)

## Loss Function
#	Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))

## Optimization and Initialization
#	Initialize variables
init = tf.global_variables_initializer()
sess.run(init)  #or: sess.run(B.initializer)
#	Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

## Train Model
loss_batch = []
#	Run loop
for i in range(100):
	rand_index = np.random.choice(100, size=batch_size)
	rand_x = np.transpose([x_vals[rand_index]])
	rand_y = np.transpose([y_vals[rand_index]])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if (i+1)%5==0:
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		print('Step #' + str(i+1) + '\tA = ' + str(sess.run(B)) + '\tLoss = ' + str(temp_loss))
		loss_batch.append(temp_loss)

accuracy_batch_____ = tf.constant(x_vals, dtype=tf.float32)
accuracy_batch_____ = tf.expand_dims(accuracy_batch_____, axis=1)
accuracy_batch_____ = sess.run( my_output, feed_dict={x_data: sess.run(accuracy_batch_____)} )
#accuracy_batch_____ = sum(x==y for x,y in zip(accuracy_batch_____, y_vals))/100.
#accuracy_batch_____ = np.mean(np.array(accuracy_batch_____) == np.array(y_vals))
accuracy_batch_____ = np.mean([ np.abs(x-y)<=1e-6  for x,y in zip(accuracy_batch_____, y_vals)])


# Plot Stochastic vs Batch Training
plt.figure()
plt.plot(range(0, 100, 5), loss_stochastic, 'b-' , label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch     , 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()

print("Accuracy of Stochastic ", np.round(accuracy_stochastic, 4))
print("Accuracy of Batch      ", np.round(accuracy_batch_____, 4))


'''
2018-02-28 21:02:47.716037: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 52.1s]
[Finished in 24.8s]

2018-02-28 21:08:58.634696: 
Chapter 2.6 	 Stochastic Training
Step #5	A = [ 1.38689542]	Loss = [ 72.70297241]
Step #10	A = [ 3.01853108]	Loss = [ 53.4449234]
Step #15	A = [ 4.26904154]	Loss = [ 33.80847931]
Step #20	A = [ 5.29814005]	Loss = [ 26.43682671]
Step #25	A = [ 6.15784979]	Loss = [ 13.1345892]
Step #30	A = [ 6.90465927]	Loss = [ 17.45701408]
Step #35	A = [ 7.40970993]	Loss = [ 3.63630199]
Step #40	A = [ 8.00195885]	Loss = [ 8.43712044]
Step #45	A = [ 8.29240799]	Loss = [ 1.41703427]
Step #50	A = [ 8.72628593]	Loss = [ 6.03347445]
Step #55	A = [ 8.93351841]	Loss = [ 0.48859617]
Step #60	A = [ 9.13013649]	Loss = [ 2.19901967]
Step #65	A = [ 9.32742882]	Loss = [ 2.09343505]
Step #70	A = [ 9.49464989]	Loss = [ 5.33907413]
Step #75	A = [ 9.58685493]	Loss = [ 0.03160714]
Step #80	A = [ 9.43920231]	Loss = [ 0.35913581]
Step #85	A = [ 9.60065746]	Loss = [ 0.52991134]
Step #90	A = [ 9.74435139]	Loss = [ 1.13328588]
Step #95	A = [ 9.81785965]	Loss = [ 0.02740372]
Step #100	A = [ 9.73111916]	Loss = [ 0.03243995]
Chapter 2.6 	 Batch Training
Step #5	A = [[ 1.31299376]]	Loss = 75.3061
Step #10	A = [[ 2.90114713]]	Loss = 51.2809
Step #15	A = [[ 4.20878553]]	Loss = 35.2325
Step #20	A = [[ 5.2696991]]	Loss = 22.7136
Step #25	A = [[ 6.13378239]]	Loss = 15.08
Step #30	A = [[ 6.83361578]]	Loss = 10.2534
Step #35	A = [[ 7.40194225]]	Loss = 7.79604
Step #40	A = [[ 7.88466835]]	Loss = 6.00479
Step #45	A = [[ 8.24999809]]	Loss = 3.02895
Step #50	A = [[ 8.54876614]]	Loss = 2.95714
Step #55	A = [[ 8.79451466]]	Loss = 2.49639
Step #60	A = [[ 9.0091877]]	Loss = 1.67813
Step #65	A = [[ 9.16112041]]	Loss = 0.897072
Step #70	A = [[ 9.27885151]]	Loss = 1.22824
Step #75	A = [[ 9.38236046]]	Loss = 0.869105
Step #80	A = [[ 9.48802471]]	Loss = 1.05145
Step #85	A = [[ 9.56782913]]	Loss = 1.1546
Step #90	A = [[ 9.63324642]]	Loss = 0.823927
Step #95	A = [[ 9.67089939]]	Loss = 0.872329
Step #100	A = [[ 9.68953323]]	Loss = 0.954898
Accuracy of Stochastic  0.0
Accuracy of Batch       0.0
[Finished in 19.4s]
'''




#---------------------------------------
#		07 Combining Everything Together
#---------------------------------------





#---------------------------------------
#		08 Evaluating Models
#---------------------------------------





