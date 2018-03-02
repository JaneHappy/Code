# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/06_Neural_Networks/07_Improving_Linear_Regression/07_improving_linear_regression.ipynb

from __future__ import division
from __future__ import print_function


# Improving Linear Regression with Neural Networks (Logistic Regression)

import csv
import os.path
import requests

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops 

# reset computational graph
ops.reset_default_graph()



## Obtain and prepare data for modeling
#	 name of data file
birth_weight_file = 'birth_weight.csv'

#	 download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
	birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
	birth_file = requests.get(birthdata_url)
	birth_data = birth_file.text.split('\r\n')
	birth_header = birth_data[0].split('\t')
	birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
	with open(birth_weight_file, "w") as f:
		writer = csv.writer(f)
		writer.writerows(birth_data)
		f.close()

#	 read birth weight data into memory
birth_data = []
#	with open(birth_weight_file, newline='') as csvfile:
with open(birth_weight_file, "rb") as csvfile:
	csv_reader = csv.reader(csvfile)
	birth_header = next(csv_reader)
	for row in csv_reader:
		birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]

#	 Pull out target variable
###y_vals = np.array([x[1]   for x in birth_data])
#	 Pull out predictor variables (not id, not target, and not birthweight)
###x_vals = np.array([x[2:9] for x in birth_data])
y_vals = np.array([x[0]  for x in birth_data])
x_vals = np.array([x[2:9] for x in birth_data])  #x[1:]


#	 set for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

#	 Declare batch size
batch_size = 90

#	 Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices  = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test  = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test  = y_vals[test_indices]

#	 Normalize by column (min-max norm)
def normalize_cols(m):
	col_max = m.max(axis=0)
	col_min = m.min(axis=0)
	return (m-col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test  = np.nan_to_num(normalize_cols(x_vals_test))



## Define Tensorflow computational graph
#	 Create graph
sess = tf.Session()

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 Create variable definition
def init_variable(shape):
	return(tf.Variable(tf.random_normal(shape=shape)))

#	 Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation=True):
	linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
	#	 We separate the activation at the end because the loss function will
	#	 implement the last sigmoid necessary
	if activation:
		return(tf.nn.sigmoid(linear_layer))
	else:
		return(linear_layer)


#	 First logistic layer (7 inputs to 7 hidden nodes)
A1 = init_variable(shape=[7, 14])
b1 = init_variable(shape=[14])
logistic_layer1 = logistic(x_data, A1, b1) 				#[None,14]

#	 Second logistic layer (7 hidden inputs to 5 hidden nodes)
A2 = init_variable(shape=[14, 5])
b2 = init_variable(shape=[5])
logistic_layer2 = logistic(logistic_layer1, A2, b2) 	#[None,5]

#	 Final output layer (5 hidden nodes to 1 output)
A3 = init_variable(shape=[5, 1])
b3 = init_variable(shape=[1])
final_output = logistic(logistic_layer2, A3, b3, activation=False) 	#[None,1]
#final_output = logistic(logistic_layer2, A3, b3)

#	 Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y_target)) 	#[None,1] -> a number

#	 Declare optimizer
my_opt = tf.train.AdamOptimizer(learning_rate=0.002)
train_step = my_opt.minimize(loss)



## Train model
#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Actual Prediction
prediction  = tf.round(tf.nn.sigmoid(final_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)  #dtype=
accuracy = tf.reduce_mean(predictions_correct)
'''
prediction 					#[None,1]
predictions_correct 		#[None,1]
accuracy 					#a number
'''

#	 Training loop
loss_vec  = []
train_acc = []
test_acc  = []
for i in range(1500):
	rand_index = np.random.choice(len(x_vals_train), size=batch_size)
	rand_x = x_vals_train[rand_index] 						#[batch_size,7]
	rand_y = np.transpose([y_vals_train[rand_index]]) 		#[batch_size,1]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
	train_acc.append(temp_acc_train)
	temp_acc_test  = sess.run(accuracy, feed_dict={x_data: x_vals_test , y_target: np.transpose([y_vals_test ])})
	test_acc.append(temp_acc_test)
	if (i+1)%150==0:
		#print('Loss = ' + str(temp_loss))
		print('Step #'+str(i+1), '\t Temp loss =', temp_loss, '\t trn acc =', temp_acc_train, '\t tst acc =', temp_acc_test)



plt.figure(figsize=(12, 5))
plt.subplot(121)
#	 %matplotlib inline
#	 Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title( 'Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
#plt.show()

plt.subplot(122)
#	 Plot train and test accuracy
plt.plot(train_acc, 'k-' , label='Train Set Accuracy')
plt.plot(test_acc , 'r--', label='Test Set Accuracy' )
plt.title( 'Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()



'''
exe-1th
2018-03-02 14:41:16.404505: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step # 150 	 Temp loss = -78.9875 	 trn acc = 0.0 	 tst acc = 0.0
Step # 300 	 Temp loss = -111.046 	 trn acc = 0.0 	 tst acc = 0.0
Step # 450 	 Temp loss = -137.254 	 trn acc = 0.0 	 tst acc = 0.0
Step # 600 	 Temp loss = -165.492 	 trn acc = 0.0 	 tst acc = 0.0
Step # 750 	 Temp loss = -189.069 	 trn acc = 0.0 	 tst acc = 0.0
Step # 900 	 Temp loss = -252.718 	 trn acc = 0.0 	 tst acc = 0.0
Step #1050 	 Temp loss = -288.09 	 trn acc = 0.0 	 tst acc = 0.0
Step #1200 	 Temp loss = -320.824 	 trn acc = 0.0 	 tst acc = 0.0
Step #1350 	 Temp loss = -353.003 	 trn acc = 0.0 	 tst acc = 0.0
Step #1500 	 Temp loss = -388.412 	 trn acc = 0.0 	 tst acc = 0.0
[Finished in 134.7s]

exe-2nd
layer1, [7,7], [7]; 	layer2, [7,.]
exe-3rd
final_output		activation=True

'''


print("y_vals_test       ", y_vals_test)
###print("y_vals_test eval  ", sess.run(prediction, feed_dict={x_data: x_vals_test , y_target: np.transpose([y_vals_test ])}))
###print("y_vals_test check?", sess.run(predictions_correct, feed_dict={x_data: x_vals_test , y_target: np.transpose([y_vals_test ])}))
print("y_vals_test eval  ", sess.run(tf.squeeze(prediction), feed_dict={x_data: x_vals_test , y_target: np.transpose([y_vals_test ])}))
print("y_vals_test check?", sess.run(tf.squeeze(predictions_correct), feed_dict={x_data: x_vals_test , y_target: np.transpose([y_vals_test ])}))

'''
exe-4th
2018-03-02 14:49:34.979290: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #150 	 Temp loss = -78.9875 	 trn acc = 0.0 	 tst acc = 0.0
Step #300 	 Temp loss = -111.046 	 trn acc = 0.0 	 tst acc = 0.0
Step #450 	 Temp loss = -137.254 	 trn acc = 0.0 	 tst acc = 0.0
Step #600 	 Temp loss = -165.492 	 trn acc = 0.0 	 tst acc = 0.0
Step #750 	 Temp loss = -189.069 	 trn acc = 0.0 	 tst acc = 0.0
Step #900 	 Temp loss = -252.718 	 trn acc = 0.0 	 tst acc = 0.0
Step #1050 	 Temp loss = -288.09 	 trn acc = 0.0 	 tst acc = 0.0
Step #1200 	 Temp loss = -320.824 	 trn acc = 0.0 	 tst acc = 0.0
Step #1350 	 Temp loss = -353.003 	 trn acc = 0.0 	 tst acc = 0.0
Step #1500 	 Temp loss = -388.412 	 trn acc = 0.0 	 tst acc = 0.0
y_vals_test        [ 29.  31.  27.  22.  32.  24.  20.  25.  19.  19.  19.  29.  22.  19.  20.
  22.  15.  16.  28.  17.  28.  25.  26.  14.  35.  19.  19.  45.  21.  26.
  18.  15.  32.  27.  21.  24.  16.  24.]
y_vals_test eval   [[ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]
 [ 1.]]
y_vals_test check? [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
[Finished in 34.9s]



exe-5th
2018-03-02 14:53:24.343563: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #150 	 Temp loss = 0.593109 	 trn acc = 0.688742 	 tst acc = 0.684211
Step #300 	 Temp loss = 0.566727 	 trn acc = 0.688742 	 tst acc = 0.684211
Step #450 	 Temp loss = 0.528934 	 trn acc = 0.715232 	 tst acc = 0.710526
Step #600 	 Temp loss = 0.451931 	 trn acc = 0.834437 	 tst acc = 0.763158
Step #750 	 Temp loss = 0.415418 	 trn acc = 0.84106 	 tst acc = 0.842105
Step #900 	 Temp loss = 0.331274 	 trn acc = 0.854305 	 tst acc = 0.815789
Step #1050 	 Temp loss = 0.331291 	 trn acc = 0.887417 	 tst acc = 0.789474
Step #1200 	 Temp loss = 0.315202 	 trn acc = 0.927152 	 tst acc = 0.710526
Step #1350 	 Temp loss = 0.260282 	 trn acc = 0.940397 	 tst acc = 0.763158
Step #1500 	 Temp loss = 0.225309 	 trn acc = 0.97351 	 tst acc = 0.763158
[Finished in 38.3s]


exe-6th
2018-03-02 14:55:22.510073: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #150 	 Temp loss = 0.593109 	 trn acc = 0.688742 	 tst acc = 0.684211
Step #300 	 Temp loss = 0.566727 	 trn acc = 0.688742 	 tst acc = 0.684211
Step #450 	 Temp loss = 0.528934 	 trn acc = 0.715232 	 tst acc = 0.710526
Step #600 	 Temp loss = 0.451931 	 trn acc = 0.834437 	 tst acc = 0.763158
Step #750 	 Temp loss = 0.415418 	 trn acc = 0.84106  	 tst acc = 0.842105
Step #900 	 Temp loss = 0.331274 	 trn acc = 0.854305 	 tst acc = 0.815789
Step #1050 	 Temp loss = 0.331291 	 trn acc = 0.887417 	 tst acc = 0.789474
Step #1200 	 Temp loss = 0.315202 	 trn acc = 0.927152 	 tst acc = 0.710526
Step #1350 	 Temp loss = 0.260282 	 trn acc = 0.940397 	 tst acc = 0.763158
Step #1500 	 Temp loss = 0.225309 	 trn acc = 0.97351  	 tst acc = 0.763158
y_vals_test        [ 1.  0.  1.  0.  0.  1.  1.  1.  1.  1.  0.  0.  1.  0.  1.  0.  1.  0.
  0.  1.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.]
y_vals_test eval   [ 1.  1.  1.  0.  0.  1.  1.  1.  1.  1.  0.  0.  1.  0.  1.  0.  1.  0.
  0.  1.  0.  0.  0.  1.  0.  0.  1.  0.  1.  1.  1.  1.  1.  1.  1.  0.
  0.  0.]
y_vals_test check? [ 1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
  1.  1.  1.  1.  1.  1.  1.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1.
  1.  1.]
[Finished in 49.1s]
'''


