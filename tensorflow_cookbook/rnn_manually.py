# coding: utf-8
# RNN
# ref: https://www.jianshu.com/p/1db917db512b




from __future__ import division
from __future__ import print_function

import numpy as np 

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import tensorflow as tf 


# initial
num_epochs 					= 6 #100
total_series_length 		= 50000
truncated_backprop_length 	= 15
state_size 					= 4
num_classes 				= 2
echo_step 					= 3
batch_size 					= 5
num_batches 				= total_series_length//batch_size//truncated_backprop_length
'''
50000/5  = 10000.0
50000//5 = 10000
	50000//5 /15  = 666.666--
	50000//5 //15 = 666
	50000/5 //15 = 666.0
	50000/5 /15  = 666.666--
		10000.0 or 10000.00 //15 = 666.0
		10000.00 //15.0 = 666.0
'''


# generate data
def generateData():
	x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5])) 					#[total_series_length,] i.e., [50000,]
	y = np.roll(x, echo_step) 																#[total_series_length,] i.e., [50000,]
	y[0: echo_step] = 0

	x = x.reshape((batch_size, -1)) 								#[batch_size,total_series_length/batch_size] i.e., [5,10000]
	y = y.reshape((batch_size, -1)) 								#[batch_size,total_series_length/batch_size] i.e., [5,10000]

	return (x, y)


# construct model

## placeholders and variables
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length]) 	#[.,.] i.e., [5,15]
batchY_placeholder = tf.placeholder(tf.int32,   [batch_size, truncated_backprop_length]) 	#[.,.] i.e., [5,15]

init_state = tf.placeholder(tf.float32, [batch_size, state_size]) 							#[.,.] i.e., [5,4]

W1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32) 				#[state_size+1,state_size] i.e., [4+1,4]
b1 = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32) 								#[1, state_size] 		   i.e., [1,4]

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32) 				#[state_size,num_classes]  i.e., [4,2]
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32) 								#[1, num_classes] 		   i.e., [1,2]

## modeling (graph)
inputs_series = tf.unstack(batchX_placeholder, axis=1) 			#truncated_backprop_length=15 items, [batch_size,]=[5,]
labels_series = tf.unstack(batchY_placeholder, axis=1) 			#truncated_backprop_length=15 items, [batch_size,]=[5,]

current_state = init_state 										#[batch_size,state_size] = [5,4]
states_series = []
for current_input in inputs_series: 								#[batch_size,]  = [5,]
	current_input = tf.reshape(current_input, [batch_size, 1]) 		#[batch_size,1] = [5,1]
	# Increasing number of columns
	#input_and_state_concatenated = tf.concat(1, [current_input, current_state]) #bug!
	input_and_state_concatenated = tf.concat([current_input, current_state], axis=1) 	#[batch_size,1+state_size] = [5,1+4]

	# Broadcasted addition
	next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W1) + b1) 				#[batch_size,state_size] = [5,4]
	states_series.append(next_state)
	current_state = next_state


## calculate loss function
# Broadcasted addition
logits_series = [ tf.matmul(state, W2) + b2  for state in states_series]
predictions_series = [ tf.nn.softmax(logits)  for logits in logits_series]

losses = []
for logits, labels in zip(logits_series, labels_series):
	losses.append( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) )
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


## visualization, visualisation
def plot(loss_list, predictions_series, batchX, batchY):
	plt.subplot(2, 3, 1)
	plt.cla()
	plt.plot(loss_list)

	for batch_series_idx in range(5):
		one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
		single_output_series  = np.array([ (1 if out[0]<0.5 else 0)  for out in one_hot_output_series])

		plt.subplot(2, 3, batch_series_idx + 2)
		plt.cla()
		plt.axis([0, truncated_backprop_length, 0, 2])
		left_offset = range(truncated_backprop_length)
		plt.bar(left_offset, batchX[batch_series_idx, :]      , width=1, color="blue" )
		plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red"  )
		plt.bar(left_offset, single_output_series * 0.3       , width=1, color="green")

	plt.draw()
	plt.pause(0.0001)


# training
with tf.Session() as sess:
	#sess.run( tf.initialize_all_variables() )
	sess.run( tf.global_variables_initializer() )
	plt.ion()
	#plt.figure()
	plt.figure(figsize=(12, 5))
	plt.show()
	loss_list = []

	for epoch_idx in range(num_epochs):
		x, y = generateData()
		_current_state = np.zeros((batch_size, state_size))

		print("New data, epoch", epoch_idx)

		for batch_idx in range(num_batches):
			start_idx = batch_idx * truncated_backprop_length
			end_idx   = start_idx + truncated_backprop_length

			batchX = x[:, start_idx:end_idx]
			batchY = y[:, start_idx:end_idx]

			_total_loss, _train_step, _current_state, _predictions_series = sess.run(
					[total_loss, train_step, current_state, predictions_series], 
					feed_dict={batchX_placeholder: batchX, batchY_placeholder: batchY, init_state: _current_state}
				)

			loss_list.append(_total_loss)

			if batch_idx % 100 == 0:
				print("\t", end="")
				print("Step", batch_idx, "Loss", _total_loss)
				plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()






'''
2018-03-29 22:51:41.307708: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
New data, epoch 0
	Step 0 Loss 0.712759
/usr/local/lib/python2.7/dist-packages/matplotlib/backend_bases.py:2453: MatplotlibDeprecationWarning: Using default event loop until function specific to this GUI is implemented
  warnings.warn(str, mplDeprecation)
	Step 100 Loss 0.0210051
	Step 200 Loss 0.00631538
	Step 300 Loss 0.0041073
	Step 400 Loss 0.00285551
	Step 500 Loss 0.00206072
	Step 600 Loss 0.00242921
New data, epoch 1
	Step 0 Loss 0.128845
	Step 100 Loss 0.00135211
	Step 200 Loss 0.00116435
	Step 300 Loss 0.00086505
	Step 400 Loss 0.000788335
	Step 500 Loss 0.000742408
	Step 600 Loss 0.000583629
New data, epoch 2
	Step 0 Loss 0.14706
	Step 100 Loss 0.000650982
	Step 200 Loss 0.000554907
	Step 300 Loss 0.000535491
	Step 400 Loss 0.000478117
	Step 500 Loss 0.00046557
	Step 600 Loss 0.000407823
New data, epoch 3
	Step 0 Loss 0.245636
	Step 100 Loss 0.00112979
	Step 200 Loss 0.000929947
	Step 300 Loss 0.000846381
	Step 400 Loss 0.000590237
	Step 500 Loss 0.000814914
	Step 600 Loss 0.000549968
New data, epoch 4
	Step 0 Loss 0.247092
	Step 100 Loss 0.000673212
	Step 200 Loss 0.00063751
invalid command name "139972589075824idle_draw"
    while executing
"139972589075824idle_draw"
    ("after" script)
	Step 300 Loss 0.000524769
	Step 400 Loss 0.00046224
	Step 500 Loss 0.00051787
	Step 600 Loss 0.000517116
New data, epoch 5
	Step 0 Loss 0.239282
	Step 100 Loss 0.000430663
	Step 200 Loss 0.000442682
[Cancelled]


2018-03-29 22:53:46.779714: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
New data, epoch 0
	Step 0 Loss 0.762416
/usr/local/lib/python2.7/dist-packages/matplotlib/backend_bases.py:2453: MatplotlibDeprecationWarning: Using default event loop until function specific to this GUI is implemented
  warnings.warn(str, mplDeprecation)
	Step 100 Loss 0.70006
	Step 200 Loss 0.660831
	Step 300 Loss 0.255059
	Step 400 Loss 0.0182017
	Step 500 Loss 0.00895085
	Step 600 Loss 0.00665291
New data, epoch 1
	Step 0 	 Loss 0.164789
	Step 100 Loss 0.00433559
	Step 200 Loss 0.00317143
	Step 300 Loss 0.00303984
	Step 400 Loss 0.00327533
	Step 500 Loss 0.0027105
	Step 600 Loss 0.00220135
New data, epoch 2
	Step 0 	 Loss 0.140049
	Step 100 Loss 0.00198972
	Step 200 Loss 0.00169418
	Step 300 Loss 0.00157706
	Step 400 Loss 0.00140513
	Step 500 Loss 0.00131931
	Step 600 Loss 0.00119959
New data, epoch 3
	Step 0 	 Loss 0.165966
	Step 100 Loss 0.00122363
	Step 200 Loss 0.00102146
	Step 300 Loss 0.00098852
	Step 400 Loss 0.000905547
	Step 500 Loss 0.000846861
	Step 600 Loss 0.000907549
New data, epoch 4
	Step 0 	 Loss 0.253389
	Step 100 Loss 0.000819925
	Step 200 Loss 0.000687365
	Step 300 Loss 0.000572223
	Step 400 Loss 0.000750771
	Step 500 Loss 0.000671128
	Step 600 Loss 0.000690725
New data, epoch 5
	Step 0 	 Loss 0.245339
	Step 100 Loss 0.000670487
	Step 200 Loss 0.000646358
	Step 300 Loss 0.00059722
	Step 400 Loss 0.000453031
	Step 500 Loss 0.000543338
	Step 600 Loss 0.000551091
[Finished in 181.3s]
'''








