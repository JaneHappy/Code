# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/06_Neural_Networks/04_Single_Hidden_Layer_Network/04_single_hidden_layer_network.ipynb

from __future__ import division
from __future__ import print_function


# Implementing a one-layer Neural Network
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets
import tensorflow as tf 
from tensorflow.python.framework import ops 

ops.reset_default_graph()

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3]   for x in iris.data])


#	 Create graph session 
sess = tf.Session()

#	 make results reproducible
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

#	 Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices  = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_trn = x_vals[train_indices]
x_vals_tst = x_vals[test_indices ]
y_vals_trn = y_vals[train_indices]
y_vals_tst = y_vals[test_indices ]


#	 Normalize by column (min-max norm)
def normalize_cols(m):
	col_max = m.max(axis=0)
	col_min = m.min(axis=0)
	return (m-col_min) / (col_max - col_min)

x_vals_trn = np.nan_to_num(normalize_cols(x_vals_trn))
x_vals_tst = np.nan_to_num(normalize_cols(x_vals_tst))



#	 Declare batch size
batch_size = 50

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 Create variables for both NN layers
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes])) # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1])) # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output

#	 Declare model operations
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output  = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

'''
tf.matmul(x_data, A1) 		#[None, hidden_layer_nodes]  #x_data * A1 [num_feat,hidden_layer_nodes]
tf.add(., b1) 				#[None, hidden_layer_nodes]  #(x_data*A1)+b1  #同行对应元素相乘；在同一个隐节点，原来所有特征的偏置项一样
tf.nn.relu(.) 				#[None, hidden_layer_nodes]  #relu(xA+b)

tf.matmul(hidden_output, A2) 	#[None,1]  #relu(xA+b) * A2
tf.add(., b2) 					#[None,1]
tf.nn.relu(.) 					#[None,1]
'''


#	 Declare loss function (MSE)
loss = tf.reduce_mean(tf.square(y_target - final_output)) 	#[None,1] -> a number

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Training loop
loss_vec = []
test_loss = []
for i in range(500):
	rand_index = np.random.choice(len(x_vals_trn), size=batch_size)
	rand_x = x_vals_trn[rand_index] 					#[batch_size,3]
	rand_y = np.transpose([y_vals_trn[rand_index]]) 	#[batch_size,1]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(np.sqrt(temp_loss))

	test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_tst, y_target: np.transpose([y_vals_tst])})
	test_loss.append(np.sqrt(test_temp_loss))
	if (i+1)%50==0:
		print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss) + '. Test Loss = ' + str(test_temp_loss))


#	 %matplotlib inline
#	 Plot loss (MSE) over time
plt.plot(loss_vec,  'k-',  label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss' )
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()


'''
2018-03-02 11:08:32.789433: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Generation:  50. Loss = 0.527901 . Test Loss = 0.259871
Generation: 100. Loss = 0.228715 . Test Loss = 0.14448
Generation: 150. Loss = 0.179773 . Test Loss = 0.100637
Generation: 200. Loss = 0.107899 . Test Loss = 0.0961823
Generation: 250. Loss = 0.240029 . Test Loss = 0.0956866
Generation: 300. Loss = 0.15324  . Test Loss = 0.0873632
Generation: 350. Loss = 0.165901 . Test Loss = 0.0828233
Generation: 400. Loss = 0.0957248. Test Loss = 0.0827632
Generation: 450. Loss = 0.121014 . Test Loss = 0.0859689
Generation: 500. Loss = 0.129494 . Test Loss = 0.080004
[Finished in 44.1s]
'''
