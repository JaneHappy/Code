# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/08_Implementing_Logistic_Regression

from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt 
import numpy as np 

import requests
import os.path
import csv

import tensorflow as tf 
from tensorflow.python.framework import ops


ops.reset_default_graph()
#	 Create graph
sess = tf.Session()

## Obtain and prepare data for modeling
#	 name of data file
birth_weight_file = 'birth_weight.csv'

# download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
	birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/' + \
		'raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
	birth_file   = requests.get(birthdata_url)
	birth_data   = birth_file.text.split('\r\n')
	birth_header = birth_data[0].split('\t')
	birth_data   = [ [ float(x)  for x in y.split('\t') if len(x)>=1]  for y in birth_data[1:] if len(y)>=1]
	with open(birth_weight_file, "w") as f:
		writer = csv.writer(f)
		writer.writerow(birth_header)
		writer.writerows(birth_data)
		f.close()

'''
2018-03-02 01:16:32.435235: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 8.2s]

2018-03-02 01:22:51.584208: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 4.3s]
'''


#	 read birth weight data into memory
birth_data = []
#with open(birth_weight_file, newline='') as csvfile:
with open(birth_weight_file, "rb") as csvfile:
	csv_reader   = csv.reader(csvfile)
	birth_header = next(csv_reader)
	for row in csv_reader:
		birth_data.append(row)

birth_data = [[ float(x)  for x in row]  for row in birth_data]

#	 Pull out target variable
y_vals = np.array([ x[0]  for x in birth_data])
#	 Pull out predictor variables (not id, not target, and not birthweight)
x_vals = np.array([ x[1:8]  for x in birth_data])

#	 set for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test__indices = np.array(list( set(range(len(x_vals))) - set(train_indices) ))
x_vals_trn = x_vals[train_indices]
x_vals_tst = x_vals[test__indices]
y_vals_trn = y_vals[train_indices]
y_vals_tst = y_vals[test__indices]

#	 Normalize by column (min-max norm)
def normalize_cols(m):
	col_max = m.max(axis=0)
	col_min = m.min(axis=0)
	return (m - col_min) / (col_max - col_min)

x_vals_trn = np.nan_to_num(normalize_cols(x_vals_trn))
x_vals_tst = np.nan_to_num(normalize_cols(x_vals_tst))
'''
?
先归一化再划分？
不，先划分再归一化，因为分类时只知道当前集合。
'''


## Define Tensorflow computational graph
#	 Declare batch size
batch_size = 25

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[7, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

#	 Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b) 	#[None,1]

#	 Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
'''
tf.nn.sigmoid_cross_entropy_with_logits 	#[None,1]   ?
tf.reduce_mean()							#a number 
'''

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)


## Train model
#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Actual Prediction
prediction  = tf.round(tf.sigmoid(model_output)) 							#[None,1]
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32) 	#[None,1]  #dtype=
accuracy    = tf.reduce_mean(predictions_correct)

#	 Training loop
loss_vec  = []
train_acc = []
test_acc  = []
for i in range(1500):
	index  = np.random.choice(len(x_vals_trn), size=batch_size)
	rand_x = x_vals_trn[index] 					#[batch_size,7]
	rand_y = np.transpose([ y_vals[index] ]) 	#[batch_size,1]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	temp_acc_trn = sess.run(accuracy, feed_dict={x_data: x_vals_trn, y_target: np.transpose([y_vals_trn]) })
	train_acc.append(temp_acc_trn)
	temp_acc_tst = sess.run(accuracy, feed_dict={x_data: x_vals_tst, y_target: np.transpose([y_vals_tst]) })
	test_acc.append( temp_acc_tst)

	if (i+1)%300==0:
		print('Step #'+str(i+1) + '\t Loss = '+str(temp_loss) + '\t acc_trn = '+str(np.round(temp_acc_trn,4)) + '\t acc_tst = '+str(np.round(temp_acc_tst,4)) )


plt.figure(figsize=(12,4.5))
plt.subplot(121)
## Display model performance
#	 %matplotlib inline
#	 Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
#plt.show()

plt.subplot(122)
#	 Plot train and test accuracy
plt.plot(train_acc, 'r-' , label='Train Set Accuracy')
plt.plot(test_acc , 'g--', label='Test  Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()






