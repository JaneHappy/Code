# coding: utf-8

from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

num_epochs  = 20 #100
total_series_length = 500 #50000
truncated_backprop_length = 15
state_size  = 4
num_classes = 2
echo_step   = 3
batch_size  = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


'''
>>> tf.__version__
'1.4.1'
>>> tf.__version__[0]
'1'
>>> from __future__ import division
>>> 500/21
23.80952380952381
>>> 500//21
23
>>> 
'''


# Generate Data
def generateData():
	x = np.array(np.random.choice(2, size=total_series_length, p=[0.5, 0.5]))
	y = np.roll(x, echo_step)
	y[0:echo_step] = 0
	x = x.reshape((batch_size, -1))
	y = y.reshape((batch_size, -1))
	return(x, y)

'''
>>> np.random.choice(2, 10, p=[0.5, 0.5])
array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
>>> np.random.choice(2, size=10, p=[0.5, 0.5])
array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
>>> 
>>> a = np.random.choice(2, size=10, p=[0.5, 0.5])
>>> a
array([0, 0, 0, 1, 1, 1, 0, 1, 1, 1])
>>> np.roll(a, 3)
array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
>>> a = np.random.choice(2, size=10, p=[0.5, 0.5])
>>> a
array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
>>> np.roll(a, 3)
array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])
>>> 
'''

# Construct Model
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32  , [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b1 = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size  , num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Construct Graph
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

current_state = init_state
states_series = []
for current_input in inputs_series:
	current_input = tf.reshape(current_input, [batch_size, 1])
	# Increasing number of columns
	input_and_state_concatenated = tf.concat(1, [current_input, current_state])
	# Broadcasted addition
	next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W1) + b1)
	states_series.append(next_state)
current_state = next_state

'''
batchX_placeholder 		#[batch_size, truncated_backprop_length] -> [5,15]
batchY_placeholder		#[batch_size, truncated_backprop_length] -> [5,15]
init_state 				#[batch_size, state_size] 				 -> [5,4]
W1 						#[state_size+1, state_size] 			 -> [5,4]
b1 						#[1, state_size] 						 -> [1,4]
W2 						#[state_size, num_classes] 				 -> [4,2]
b2 						#[1, num_classes] 						 -> [1,2]
inputs_series 			#[batch_size,] truncated_backprop_length -> [5,] 15 items
labels_series 			#[batch_size,] truncated_backprop_length -> [5,] 15 items

	current_state 						#[batch_size, state_size] 		 -> [5,4]
		current_input 					#[batch_size, ] 				 -> [5,]
		current_input 					#[batch_size, 1] 				 -> [5,1]
		input_and_state_concatenated 	#[batch_size, 1+state_size] 	 -> [5,5]
			tf.matmul(.,.) 					#[batch_size, state_size] 	 -> [5,4]
			tf.add(.,.) 					#[batch_size, state_size]
			tf.tanh(.) 						#[batch_size, state_size]
		next_state 						#[batch_size, state_size] 		 -> [5,4]
	states_series 						#[batch_size, state_size], truncated_backprop_length -> [5,4], 15 items
	current_state 						#[batch_size, state_size]
'''


# Softmax loss
# Broadcasted addition
logits_series = [ tf.matmul(state, W2) + b2  for state in states_series]
predictions_series = [ tf.nn.softmax(logits)  for logits in logits_series]

losses = []
for logits, labels in zip(logits_series, labels_series):
	losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

'''
states_series 			#[batch_size, state_size], truncated_backprop_length -> [5,4], 15 items
labels_series 			#[batch_size,] truncated_backprop_length -> [5,] 15 items
W2 						#[state_size, num_classes] 				 -> [4,2]
b2 						#[1, num_classes] 						 -> [1,2]

	state 					#[batch_size, state_size] 			 -> [5,4]
	tf.matmul()				#[batch_size, num_classes] 			 -> [5,2]
	tf.add() 				#[batch_size, num_classes] 			 -> [5,2]
		logits_series 			#[batch_size, num_classes], truncated_backprop_length 
zip(logits_series, labels_series)
	logits 					#[batch_size, num_classes]
	labels 					#[batch_size,]
	tf.nn.sparse..(.) 		#[batch_size,]
		losses 				#[batch_size,] truncated_backprop_length -> [5,] 15 items
'''









'''
References:
https://www.jianshu.com/p/1db917db512b

'''
