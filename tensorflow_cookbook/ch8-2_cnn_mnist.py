# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/08_Convolutional_Neural_Networks/02_Intro_to_CNN_MNIST/02_introductory_cnn.ipynb

from __future__ import division
from __future__ import print_function


# Introductory CNN Model: MNIST Digits
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

'''
[Finished in 11.3s]
'''

ops.reset_default_graph()
#	 Start a graph session
sess = tf.Session()

#	 Load data
#data_dir = "temp"
data_dir = ''
mnist = read_data_sets(data_dir)

'''
2018-03-02 17:45:19.402616: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Extracting train-images-idx3-ubyte.gz
Extracting train-labels-idx1-ubyte.gz
Extracting t10k-images-idx3-ubyte.gz
Extracting t10k-labels-idx1-ubyte.gz
[Finished in 25.8s]

>>> from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
>>> mnist = read_data_sets('')
Extracting train-images-idx3-ubyte.gz
Extracting train-labels-idx1-ubyte.gz
Extracting t10k-images-idx3-ubyte.gz
Extracting t10k-labels-idx1-ubyte.gz
>>> mnist
Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2d66e73190>, validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2d4a0774d0>, test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2d488285d0>)
>>> mnist.train
<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2d66e73190>
>>> mnist.train.images.shape
(55000, 784)
>>> mnist.train.labels.shape
(55000,)
>>> mnist.test.images.shape
(10000, 784)
>>> mnist.validation.images.shape
(5000, 784)
>>> 

>>> mnist.train.images[:3]
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)
>>> mnist.train.images[:3].shape
(3, 784)
>>> 
'''



# Convert images into 28x28 (they are downloaded as 1x784)
trn_xdata = np.array([np.reshape(x, (28, 28))  for x in mnist.train.images])
tst_xdata = np.array([np.reshape(x, (28, 28))  for x in mnist.test.images ])

# Convert labels into one-hot encoded vectors
trn_labels = mnist.train.labels
tst_labels = mnist.test.labels 

'''
mnist.train.images 	#[55000, 784]
	[784,] -> [28,28]
	np.reshape(x, .) 		#[28,28]
	[np.reshape(x,.) ..] 	#[np.array([28,28])  55000]
trn_xdata 			#[55000, 28,28]
tst_xdata 			#[10000, 28,28]
trn_labels 			#[55000,]
tst_labels 			#[10000,]
'''


# Set model parameters
batch_size      = 100
learning_rate   = 0.005
evaluation_size = 500
image_height    = trn_xdata[0].shape[0]
image_width     = tst_xdata[0].shape[1]
target_size     = max(trn_labels) + 1
num_channels    = 1 # greyscale = 1 channel
generations     = 500
eval_every      = 5
conv1_features  = 25
conv2_features  = 50
filter_size1    = 4 #my
filter_size2    = 4 #my
max_pool_size1  = 2 # NxN window for 1st max pool layer
max_pool_size2  = 2 # NxN window for 2nd max pool layer
fully_connected_size1 = 100

'''
________
|      |
|      | height
|______|
 width
'''

x_input_shape = (batch_size, image_height, image_width, num_channels)
x_input       = tf.placeholder(tf.float32, shape=x_input_shape) 	#[batch_size, image_height, image_width, num_channels]
y_target      = tf.placeholder(tf.int32,   shape=(batch_size))  	#[batch_size,]

eval_input_shape = (evaluation_size, image_height, image_width, num_channels)
eval_input       = tf.placeholder(tf.float32, shape=eval_input_shape) 	#[evaluation_size, .,.,.]
eval_target      = tf.placeholder(tf.int32,   shape=(evaluation_size)) 	#[evaluation_size]



# Convolutional layer variables
#conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_weight = tf.Variable(tf.truncated_normal([filter_size1, filter_size1, num_channels, conv1_features], 
												stddev=0.1, dtype=tf.float32))
conv1_bias   = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

#conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([filter_size2, filter_size2, conv1_features, conv2_features], 
												stddev=0.1, dtype=tf.float32))
conv2_bias   = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

# fully connected variables
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
resulting_width  = image_width  // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_height * resulting_width * conv2_features
full1_weight     = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
full1_bias       = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight     = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
full2_bias       = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

'''
// in python3 : 取整除 - 返回商的整数部分


>>> a = tf.zeros(shape=[1,4,4,2])
>>> s = tf.Session()
2018-03-02 18:37:26.965422: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
>>> s.run(a)
array([[[[ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.]],

        [[ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.]],

        [[ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.]],

        [[ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.],
         [ 0.,  0.]]]], dtype=float32)
>>> b = tf.constant([2.,3])
>>> s.run(b)
array([ 2.,  3.], dtype=float32)
>>> s.run(tf.nn.bias_add(a, b))
array([[[[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]],

        [[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]],

        [[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]],

        [[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]]]], dtype=float32)
>>> s.run(tf.add(a, b))
array([[[[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]],

        [[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]],

        [[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]],

        [[ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.],
         [ 2.,  3.]]]], dtype=float32)


>>> d = tf.constant(range(32))
>>> s.run(d)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], dtype=int32)
>>> d1=tf.reshape(d, [2,4,4])
>>> s.run(d1)
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]],

       [[16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31]]], dtype=int32)
>>> d2 = tf.expand_dims(d1,0)
>>> d2
<tf.Tensor 'ExpandDims_1:0' shape=(1, 2, 4, 4) dtype=int32>
>>> s.run(d2)
array([[[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23],
         [24, 25, 26, 27],
         [28, 29, 30, 31]]]], dtype=int32)
>>> s.run(tf.reshape(d2, [1,32]))
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]], dtype=int32)
>>> 



'''


'''
ERROR!!!


conv1 		#[batch_size, out_h1, out_w1, conv1_features]
	input_data 		#[batch_size, image_height, image_width, num_channels]
	conv1_weight 	#[4, 4, num_channels, conv1_features]
					#[filter_size1, filter_size1, num_channels, conv1_features]
	strides 		#[1, 1, 1, 1]
		(1) filter.1 	-> 	[filter_size1*filter_size1*num_channels, conv1_features]
		(2) patch.1 	-> 	[batch_size, out_h1, out_w1, filter_size1*filter_size1*num_channels]
		(3) 			-> 	[batch_size, out_h1, out_w1, conv1_features]
		(i ) 	out_h1 	= ceil( image_height / strides[1] ) = ceil(image_height/1) = image_height
		(ii) 	out_w1 	= ceil( image_width  / strides[2] ) = ceil(image_width /1) = image_width

relu1 		#[batch_size, out_h1, out_w1, conv1_features]
	conv1_bias 		#[conv1_features,]

max_pool1 	#[batch_size, pool_h1, pool_w1, conv1_features]
	relu1 			#[batch_size, out_h1, out_w1, conv1_features]
	ksize 			#[1, max_pool_size1, max_pool_size1, 1]
	strides 		#[1, max_pool_size1, max_pool_size1, 1]
	(iii) ? 	= [batch_size, ceil(out_h1/max_pool_size1)+1, ceil(out_w1/max_pool_size1)+1, conv1_features]


conv2 		#[batch_size, out_h2, out_w2, conv2_features]
	max_pool1 		#[batch_size, pool_h1, pool_w1, conv1_features]
	conv2_weight 	#[4, 4, conv1_features, conv2_features]
					#[filter_size2, filter_size2, conv1_features, conv2_features]
	strides 		#[1, 1, 1, 1]
		(1) filter.2 	-> 	[filter_size2*filter_size2*conv1_features, conv2_features]
		(2) patch.2 	-> 	[batch_size, out_h2, out_w2, filter_size2*filter_size2*conv1_features]
		(3) 			-> 	[batch_size, out_h2, out_w2, conv2_features]
		(i ) 	out_h2 		= ceil( pool_h1 / strides[1] ) = ceil(pool_h1/1)
		(ii) 	out_w2 		= ceil( pool_w1 / strides[2] ) = ceil(pool_w1/1)

relu2 		#[batch_size, out_h2, out_w2, conv2_features]
	conv2 			#
	conv2_bias 		#[conv2_features,]

max_pool2 	#[batch_size, pool_h2, pool_w2, conv2_features]
	relu2 			#[batch_size, out_h2, out_w2, conv2_features]
	ksize 			#[1, max_pool_size2, max_pool_size2, 1]
	strides 		#[1, max_pool_size2, max_pool_size2, 1]
	(iii) ? 	 = [batch_size, ceil(out_h2/max_pool_size2)+1, ceil(out_w2/max_pool_size2)+1, conv2_features]



batch_size      = 100
image_height    = 28 		image_width     = 28
conv1_features  = 25 		conv2_features  = 50
filter_size1    = 4  		filter_size2    = 4 
max_pool_size1  = 2  		max_pool_size2  = 2 
fully_connected_size1 = 100

	out_h1  = ceil(image_height/1) = image_height = 28 
	out_w1  = ceil(image_width /1) = image_width  = 28 
	pool_h1 = ceil(out_h1/max_pool_size1)+1 = ceil(image_height/max_pool_size1)+1 = ceil(28/2)+1 = 15
	pool_w1 = ceil(out_w1/max_pool_size1)+1 = ceil(image_width /max_pool_size1)+1 = ceil(28/2)+1 = 15
		out_h2  = ceil(pool_h1/1) = pool_h1 = 15
		out_w2  = ceil(pool_w1/1) = pool_w1 = 15
		pool_h2 = ceil(out_h2/max_pool_size2)+1 = ceil(15/2)+1 = 9
		pool_w2 = ceil(out_w2/max_pool_size2)+1 = ceil(15/2)+1 = 9

	conv1 			#[100, 28, 28, 25]
	relu1 			#[100, 28, 28, 25]
	max_pool1 		#[100, 15, 15, 25] 		ceil(28/2)+1 = 14+1
		conv2 		#[100, 15, 15, 50]
		relu2 		#[100, 15, 15, 50]
		max_pool2 	#[100,  9,  9, 50]



final_conv_shape 	= [batch_size, pool_h2, pool_w2, conv2_features]
final_shape         = pool_h2 * pool_w2 * conv2_features
flat_output 		#[batch_size, final_shape]
fully_connected1 	#[]
	flat_output 		#[batch_size, final_shape]
	full1_weight 		#[full1_input_size, fully_connected_size1]
	full1_bias 			#[fully_connected_size1,]

	full2_weight 		#[fully_connected_size1, target_size]
	full2_bias 			#[target_size,]



resulting_height = image_height // (max_pool_size1 * max_pool_size2)
resulting_width  = image_width  // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_height * resulting_width * conv2_features
	resulting_height = 28//(2*2) = 7.0 = 7
	resulting_width  = 28//(2*2) = 7.0 = 7
	full1_input_size = 7*7*50 = 
			final_shape = 9*9*50 = 

'''



# Initialize Model Operations
def my_conv_net(input_data):
	# First Conv-ReLU-MaxPool Layer
	conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
	max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1], 
							   strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

	# Second Conv-ReLU-MaxPool Layer
	conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
	max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1], 
							   strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

	# Transform Output into a 1xN layer for next fully connected layer
	final_conv_shape = max_pool2.get_shape().as_list()
	final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
	flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

	# First Fully Connected Layer
	fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

	# Second Fully Connected Layer
	final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

	return(final_model_output)

model_output = my_conv_net(x_input) 			#[batch_size, target_size] 			#[batch_size,10]
test_model_output = my_conv_net(eval_input) 	#[evaluation_size, target_size] 	#[evaluation_size,10]

'''
UPDATED!!! 

image_height = image_width = 28
num_channels  			   = 1
target_size  = 9+1         = 10
conv(1/2)_features 		   = 25/50
filter_size(1/2) 		   = 4
max_pool_size(1/2) 		   = 2
fully_connected_size1      = 100
	resulting_height    = image_height // (max_pool_size1 * max_pool_size2) = 28//(2*2) = 7.0 = 7
	resulting_width     = image_width  // (max_pool_size1 * max_pool_size2) = 28//(2*2) = 7.0 = 7
	full1_input_size    = resulting_height * resulting_width * conv2_features = 7*7*50 = 2450





conv1 				##[batch_size, out_h1, out_w1, conv1_features] 									##[batch_size,28,28,25]
	input_data 			#[batch_size, image_height, image_width, num_channels] 			#[batch_size, 28, 28, 1]
	conv1_weight 		#[filter_size1, filter_size1, num_channels, conv1_features] 	#[4, 4, 1, 25]
	strides 			#[1, 1, 1, 1] 													#[1, 1, 1, 1]
		(1) filter.1  	-> [filter_size1 * filter_size1 * num_channels, conv1_features] 			##[4*4*1,25] -> [16,25]
		(2) patch.1 	-> [batch_size, out_h1, out_w1, filter_size1*filter_size1*num_channels] 	##[batch_size,?,?,16]
		(3) 	output:	   [batch_size, out_h1, out_w1, conv1_features] 							##[batch_size,?,?,25]
		(i ) 	out_h1  =  ceil(image_height/strides[1]) = ceil(image_height/1) = image_height = 28
		(ii) 	out_w1  =  ceil(image_width /strides[2]) = ceil(image_width /1) = image_width  = 28
relu1 				##[batch_size, out_h1, out_w1, conv1_features]
	conv1 				#
	conv1_bias 			#[conv1_features,]
max_pool1 			##[batch_size, out_h3, out_w3, conv1_features] 									##[batch_size,14,14,25]
	relu1 				#[batch_size, out_h1, out_w1, conv1_features] 								##[batch_size,28,28,25]
	ksize 				#[1, max_pool_size1, max_pool_size1, 1] 						#[1, 2, 2, 1]
	strides 			#[1, max_pool_size1, max_pool_size1, 1] 						#[1, 2, 2, 1]
		(iii) 	output:	   [batch_size, out_h3, out_w3, conv1_features]
		(a)		out_h3  =  ceil((out_h1 - max_pool_size1) / max_pool_size1) + 1 = ceil((28-2)/2)+1 = ceil(13)+1 = 14
		(b) 	out_w3  =  ceil((out_w1 - max_pool_size1) / max_pool_size1) + 1 = ceil((28-2)/2)+1 = ceil(13)+1 = 14
		(c) corrected
			'VALID'	out_h3  =  ceil((out_h1-max_pool_size1+1)/max_pool_size1) = ceil((28-2+1)/2) = ceil(27/2) = 14
					out_w3  =  ceil((out_w1-max_pool_szie1+1)/max_pool_size1) = ceil((28-2+1)/2) = ceil(27/2) = 14
			'SAME' 	out_h3  =  ceil(out_h1 / max_pool_size1) = ceil(28/2) = 14
					out_h3  =  ceil(out_w1 / max_pool_size1) = ceil(28/2) = 14

conv2 				###[batch_size, out_h2, out_w2, conv2_features] 								###[batch_size,14,14,50]
	max_pool1 			##[batch_size, out_h3, out_w3, conv1_features]
	conv2_weight 		#[filter_size2, filter_size2, conv1_features, conv2_features] 	#[4, 4, 25, 50]
	strides 			#[1, 1, 1, 1] 													#[1, 1, 1, 1]
		(1) filter.2 	-> [filter_size2 * filter_size2 * conv1_features, conv2_features] 			##[4*4*25,50] -> [400,50]
		(2) patch.2 	-> [batch_size, out_h2, out_w2, filter_size2*filter_size2*conv1_features] 	##[batch_size,?,?,400]
		(3) 	output:	   [batch_size, out_h2, out_w2, conv2_features] 							##[batch_size,?,?,50]
		(i ) 	out_h2  =  ceil(out_h3/strides[1]) = ceil(out_h3/1) = out_h3 = 14
		(ii) 	out_w2  =  ceil(out_w3/strides[2]) = ceil(out_w3/1) = out_w3 = 14
relu2 				###[batch_size, out_h2, out_w2, conv2_features]
	conv2 				##
	conv2_bias 			#[conv2_features,]
max_pool2 			###[batch_size, out_h4, out_w4, conv2_features] 								###[batch_size,7,7,50]
	relu2 				##[batch_size, out_h2, out_w2, conv2_features] 								##[batch_size,14,14,50]
	ksize 				#[1, max_pool_size2, max_pool_size2, 1] 						#[1, 2, 2, 1]
	strides 			#[1, max_pool_size2, max_pool_size2, 1] 						#[1, 2, 2, 1]
		(iii) 	output:	   [batch_size, out_h4, out_w4, conv2_features]
		(a) 	out_h4  =  ceil((out_h2 - max_pool_size2) / max_pool_size2) + 1 = ceil((14-2)/2)+1 = ceil(6)+1 = 7
		(b) 	out_w4  =  ceil((out_w2 - max_pool_size2) / max_pool_size2) + 1 = ceil((14-2)/2)+1 = ceil(6)+1 = 7
		(c) corrected
			'VALID' out_h4  =  ceil((out_h2-max_pool_size2+1)/max_pool_size2) = ceil((14-2+1)/2) = ceil(13/2) = 7
					out_w4  =  ceil((out_w2-max_pool_size2+1)/max_pool_size2) = ceil((14-2+1)/2) = ceil(13/2) = 7
			'SAME' 	out_h4  =  ceil(out_h2 / max_pool_size2) = ceil(14/2) = 7
					out_w4  =  ceil(out_h2 / max_pool_size2) = ceil(14/2) = 7

final_conv_shape = 		   [batch_size,7,7,50]
final_shape      =  	   7*7*50 = 2450
flat_output 		###[batch_size, final_shape] 													##[batch_size,2450]
	max_pool2 			#

fully_connected1 	####[batch_size, fully_connected_size1] 										###[batch_size,100]
	flat_output 		###[batch_size, final_shape] 												##[batch_size,2450]
	full1_weight 		#[full1_input_size, fully_connected_size1] 						#[2450,100]
	full1_bias 			#[fully_connected_size1,] 										#[100,]
final_model_output 	####[batch_size, target_size] 													###[batch_size,10]
	fully_connected1 	###[batch_size, fully_connected_size1] 										##[batch_size,100]
	full2_weight 		#[fully_connected_size1, target_size] 							#[100,10]
	full2_bias 			#[target_size,] 												#[10,]

'''



'''
>>> a = np.random.randint(10, size=(1,10,10,1))
>>> a
array([[[[8], [8], [7], [3], [3], [3], [2], [2], [3], [3]],
        [[1], [0], [0], [7], [4], [1], [6], [6], [3], [6]],
        [[4], [1], [5], [8], [9], [1], [7], [0], [7], [9]],
        [[2], [4], [9], [0], [4], [6], [8], [9], [9], [4]],
        [[3], [2], [1], [1], [9], [3], [3], [1], [2], [1]],
        [[8], [1], [8], [6], [5], [8], [5], [9], [2], [0]],
        [[6], [6], [8], [2], [2], [8], [3], [7], [8], [9]],
        [[7], [3], [4], [6], [2], [0], [7], [2], [6], [8]],
        [[4], [0], [0], [0], [1], [7], [9], [4], [1], [0]],
        [[4], [7], [9], [4], [0], [1], [4], [0], [3], [2]]]])
>>> ta = tf.constant(a)
>>> ta
<tf.Tensor 'Const_2:0' shape=(1, 10, 10, 1) dtype=int64>
>>> tf.nn.max_pool(ta, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
<tf.Tensor 'MaxPool:0' shape=(1, 5, 5, 1) dtype=int64>
>>> tf.nn.max_pool(ta, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
<tf.Tensor 'MaxPool_1:0' shape=(1, 4, 4, 1) dtype=int64>
>>> 

'VALID'  floor((10-3)/2)+1 = floor(7/2)+1 = floor(3.5)+1 = 4
'SAME'   ceil( (10-3)/2)+1 = ceil( 7/2)+1 = ceil( 3.5)+1 = 5

>>> tf.nn.max_pool(ta, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
<tf.Tensor 'MaxPool_2:0' shape=(1, 5, 5, 1) dtype=int64>
>>> tf.nn.max_pool(ta, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
<tf.Tensor 'MaxPool_3:0' shape=(1, 5, 5, 1) dtype=int64>

>>> tf.nn.max_pool(ta, ksize=[1,4,4,1], strides=[1,3,3,1], padding='VALID')
<tf.Tensor 'MaxPool_5:0' shape=(1, 3, 3, 1) dtype=int64>
>>> tf.nn.max_pool(ta, ksize=[1,4,4,1], strides=[1,3,3,1], padding='SAME')
<tf.Tensor 'MaxPool_4:0' shape=(1, 4, 4, 1) dtype=int64>

'VALID': 	floor((10-2)/2)+1 = 5 		floor((10-4)/3)+1=3
'SAME' : 	ceil( (10-2)/2)+1 = 5 		ceil( (10-4)/3)+1=3


corrected:
'VALID': 	ceil((10-3+1)/2)=4 	ceil((10-2+1)/2)=5 	ceil((10-4+1)/3)=ceil(7/3)=3
'SAME' : 	ceil(10/2)      =5 	ceil(10/2)      =5 	ceil(10/3)      =4

'''



# Declare Loss Function (softmax cross entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Create a prediction function
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

# Create accuracy function
def get_accuracy(logits, targets):
	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return(100. * num_correct / batch_predictions.shape[0])

# Create an optimizer
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  #momentum=0.9
train_step   = my_optimizer.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)


# Start training loop
train_loss = []
train_acc  = []
test_acc   = []
for i in range(generations):
	rand_index = np.random.choice(len(trn_xdata), size=batch_size)
	rand_x = trn_xdata[ rand_index] 				#[batch_size, 28,28]
	rand_x = np.expand_dims(rand_x, 3) 				#[batch_size, 28,28,1]
	rand_y = trn_labels[rand_index] 				#[batch_size,]
	train_dict = {x_input: rand_x, y_target: rand_y}

	sess.run(train_step, feed_dict=train_dict)
	temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict) 	#a number, [batch_size,10]
	temp_train_acc = get_accuracy(temp_train_preds, rand_y) 		#a number

	if (i+1) % eval_every == 0:
		eval_index = np.random.choice(len(tst_xdata), size=evaluation_size)
		eval_x = tst_xdata[ eval_index] 				#[evaluation_size, 28,28]
		eval_x = np.expand_dims(eval_x, 3) 				#[evaluation_size, 28,28,1]
		eval_y = tst_labels[eval_index] 				#[evaluation_size,]
		test_dict = {eval_input: eval_x, eval_target: eval_y}
		test_preds = sess.run(test_prediction, feed_dict=test_dict) 	#[evaluation_size,10]
		temp_test_acc = get_accuracy(test_preds, eval_y) 				#a number

		# Record and print results
		train_loss.append(temp_train_loss)
		train_acc.append( temp_train_acc )
		test_acc.append(  temp_test_acc  )
		acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
		acc_and_loss = [np.round(x,2) for x in acc_and_loss]
		print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))



'''
loss 			#[batch_size,] -> a number
	model_output 	#[batch_size, target_size] 		#[batch_size,10]
	y_target 		#[batch_size,] 					#[batch_size,]

prediction 		#[batch_size, target_size] 			#[batch_size,10]
test_prediction	#[evaluation_size, target_size] 	#[evaluation)_size,10]

def get_accuracy()
	logits: 	np.array, [batch_size, target_size] 	#[batch_size,10]
	targets 	np.array, [batch_size,]
		batch_predictions 		[batch_size,]
		np.equal(., targets) 	[batch_size,]
		np.sum(.) 				a number
			return: 			a number
						100.*#correct/batch_size = accuracy(%)



>>> a = np.random.randint(2, size=[3,2])
>>> a
array([[1, 1],
       [1, 1],
       [1, 0]])
>>> b = np.random.choice(3, size=(4,3))
>>> b
array([[1, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [1, 2, 1]])
>>> ta = tf.constant(a, dtype=tf.float32)
>>> tb = tf.constant(b, dtype=tf.float32)
>>> s.run(tf.nn.softmax(ta))
array([[ 0.5       ,  0.5       ],
       [ 0.5       ,  0.5       ],
       [ 0.7310586 ,  0.26894143]], dtype=float32)
>>> s.run(tf.nn.softmax(tb))
array([[ 0.57611692,  0.21194157,  0.21194157],
       [ 0.33333334,  0.33333334,  0.33333334],
       [ 0.33333334,  0.33333334,  0.33333334],
       [ 0.21194157,  0.57611692,  0.21194157]], dtype=float32)
>>> 

'''



# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)

plt.figure(figsize=(12, 5))
plt.subplot(121)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')

plt.subplot(122)
# Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-' , label='Train Set Accuracy')
plt.plot(eval_indices, test_acc , 'r--', label='Test Set Accuracy' )
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()



'''
2018-03-02 23:02:17.664053: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Extracting train-images-idx3-ubyte.gz
Extracting train-labels-idx1-ubyte.gz
Extracting t10k-images-idx3-ubyte.gz
Extracting t10k-labels-idx1-ubyte.gz
Generation # 5. Train Loss: 2.30. Train Acc (Test Acc): 10.00 (17.00)
Generation # 10. Train Loss: 2.20. Train Acc (Test Acc): 26.00 (24.20)
Generation # 15. Train Loss: 2.05. Train Acc (Test Acc): 44.00 (39.60)
Generation # 20. Train Loss: 1.89. Train Acc (Test Acc): 56.00 (61.00)
Generation # 25. Train Loss: 1.67. Train Acc (Test Acc): 64.00 (61.80)
Generation # 30. Train Loss: 1.47. Train Acc (Test Acc): 64.00 (68.00)
Generation # 35. Train Loss: 1.09. Train Acc (Test Acc): 81.00 (77.20)
Generation # 40. Train Loss: 0.85. Train Acc (Test Acc): 78.00 (82.60)
Generation # 45. Train Loss: 0.67. Train Acc (Test Acc): 80.00 (80.00)
Generation # 50. Train Loss: 0.65. Train Acc (Test Acc): 82.00 (83.40)
Generation # 55. Train Loss: 0.46. Train Acc (Test Acc): 81.00 (85.40)
Generation # 60. Train Loss: 0.56. Train Acc (Test Acc): 84.00 (87.00)
Generation # 65. Train Loss: 0.39. Train Acc (Test Acc): 85.00 (83.80)
Generation # 70. Train Loss: 0.40. Train Acc (Test Acc): 82.00 (86.20)
Generation # 75. Train Loss: 0.38. Train Acc (Test Acc): 87.00 (85.40)
Generation # 80. Train Loss: 0.29. Train Acc (Test Acc): 91.00 (86.80)
Generation # 85. Train Loss: 0.34. Train Acc (Test Acc): 86.00 (90.60)
Generation # 90. Train Loss: 0.42. Train Acc (Test Acc): 83.00 (90.00)
Generation # 95. Train Loss: 0.28. Train Acc (Test Acc): 90.00 (90.80)
Generation # 100. Train Loss: 0.20. Train Acc (Test Acc): 96.00 (93.00)
Generation # 105. Train Loss: 0.28. Train Acc (Test Acc): 90.00 (88.20)
Generation # 110. Train Loss: 0.29. Train Acc (Test Acc): 93.00 (91.40)
Generation # 115. Train Loss: 0.25. Train Acc (Test Acc): 95.00 (91.20)
Generation # 120. Train Loss: 0.21. Train Acc (Test Acc): 94.00 (92.40)
Generation # 125. Train Loss: 0.27. Train Acc (Test Acc): 92.00 (92.60)
Generation # 130. Train Loss: 0.30. Train Acc (Test Acc): 93.00 (91.40)
Generation # 135. Train Loss: 0.22. Train Acc (Test Acc): 91.00 (91.00)
Generation # 140. Train Loss: 0.26. Train Acc (Test Acc): 93.00 (92.60)
Generation # 145. Train Loss: 0.24. Train Acc (Test Acc): 93.00 (92.20)
Generation # 150. Train Loss: 0.18. Train Acc (Test Acc): 94.00 (92.00)
Generation # 155. Train Loss: 0.30. Train Acc (Test Acc): 88.00 (89.00)
Generation # 160. Train Loss: 0.24. Train Acc (Test Acc): 94.00 (92.40)
Generation # 165. Train Loss: 0.32. Train Acc (Test Acc): 92.00 (90.80)
Generation # 170. Train Loss: 0.25. Train Acc (Test Acc): 91.00 (92.00)
Generation # 175. Train Loss: 0.21. Train Acc (Test Acc): 92.00 (90.40)
Generation # 180. Train Loss: 0.24. Train Acc (Test Acc): 94.00 (91.60)
Generation # 185. Train Loss: 0.23. Train Acc (Test Acc): 93.00 (92.80)
Generation # 190. Train Loss: 0.26. Train Acc (Test Acc): 91.00 (92.60)
Generation # 195. Train Loss: 0.17. Train Acc (Test Acc): 94.00 (92.20)
Generation # 200. Train Loss: 0.13. Train Acc (Test Acc): 95.00 (91.80)
Generation # 205. Train Loss: 0.15. Train Acc (Test Acc): 93.00 (93.60)
Generation # 210. Train Loss: 0.13. Train Acc (Test Acc): 98.00 (93.40)
Generation # 215. Train Loss: 0.25. Train Acc (Test Acc): 93.00 (91.80)
Generation # 220. Train Loss: 0.19. Train Acc (Test Acc): 95.00 (93.80)
Generation # 225. Train Loss: 0.16. Train Acc (Test Acc): 96.00 (93.00)
Generation # 230. Train Loss: 0.23. Train Acc (Test Acc): 91.00 (93.80)
Generation # 235. Train Loss: 0.31. Train Acc (Test Acc): 91.00 (92.60)
Generation # 240. Train Loss: 0.18. Train Acc (Test Acc): 95.00 (94.20)
Generation # 245. Train Loss: 0.19. Train Acc (Test Acc): 93.00 (94.40)
Generation # 250. Train Loss: 0.32. Train Acc (Test Acc): 95.00 (93.80)
Generation # 255. Train Loss: 0.17. Train Acc (Test Acc): 96.00 (96.40)
Generation # 260. Train Loss: 0.21. Train Acc (Test Acc): 94.00 (94.80)
Generation # 265. Train Loss: 0.17. Train Acc (Test Acc): 95.00 (94.20)
Generation # 270. Train Loss: 0.13. Train Acc (Test Acc): 96.00 (95.40)
Generation # 275. Train Loss: 0.24. Train Acc (Test Acc): 93.00 (96.60)
Generation # 280. Train Loss: 0.18. Train Acc (Test Acc): 94.00 (93.40)
Generation # 285. Train Loss: 0.12. Train Acc (Test Acc): 96.00 (94.60)
Generation # 290. Train Loss: 0.18. Train Acc (Test Acc): 93.00 (95.60)
Generation # 295. Train Loss: 0.26. Train Acc (Test Acc): 92.00 (95.20)
Generation # 300. Train Loss: 0.14. Train Acc (Test Acc): 96.00 (94.60)
Generation # 305. Train Loss: 0.21. Train Acc (Test Acc): 94.00 (93.40)
Generation # 310. Train Loss: 0.15. Train Acc (Test Acc): 95.00 (96.00)
Generation # 315. Train Loss: 0.13. Train Acc (Test Acc): 98.00 (95.20)
Generation # 320. Train Loss: 0.18. Train Acc (Test Acc): 94.00 (94.80)
Generation # 325. Train Loss: 0.10. Train Acc (Test Acc): 98.00 (95.40)
Generation # 330. Train Loss: 0.14. Train Acc (Test Acc): 96.00 (95.00)
Generation # 335. Train Loss: 0.23. Train Acc (Test Acc): 93.00 (95.40)
Generation # 340. Train Loss: 0.25. Train Acc (Test Acc): 91.00 (94.20)
Generation # 345. Train Loss: 0.17. Train Acc (Test Acc): 93.00 (94.00)
Generation # 350. Train Loss: 0.12. Train Acc (Test Acc): 96.00 (96.40)
Generation # 355. Train Loss: 0.23. Train Acc (Test Acc): 93.00 (94.60)
Generation # 360. Train Loss: 0.08. Train Acc (Test Acc): 99.00 (92.20)
Generation # 365. Train Loss: 0.19. Train Acc (Test Acc): 94.00 (93.40)
Generation # 370. Train Loss: 0.11. Train Acc (Test Acc): 95.00 (95.40)
Generation # 375. Train Loss: 0.12. Train Acc (Test Acc): 95.00 (93.40)
Generation # 380. Train Loss: 0.12. Train Acc (Test Acc): 94.00 (95.40)
Generation # 385. Train Loss: 0.14. Train Acc (Test Acc): 95.00 (95.20)
Generation # 390. Train Loss: 0.09. Train Acc (Test Acc): 98.00 (96.20)
Generation # 395. Train Loss: 0.23. Train Acc (Test Acc): 94.00 (96.20)
Generation # 400. Train Loss: 0.12. Train Acc (Test Acc): 95.00 (96.60)
Generation # 405. Train Loss: 0.14. Train Acc (Test Acc): 96.00 (95.60)
Generation # 410. Train Loss: 0.19. Train Acc (Test Acc): 91.00 (95.00)
Generation # 415. Train Loss: 0.17. Train Acc (Test Acc): 94.00 (96.80)
Generation # 420. Train Loss: 0.19. Train Acc (Test Acc): 93.00 (93.80)
Generation # 425. Train Loss: 0.15. Train Acc (Test Acc): 94.00 (94.80)
Generation # 430. Train Loss: 0.09. Train Acc (Test Acc): 97.00 (94.60)
Generation # 435. Train Loss: 0.09. Train Acc (Test Acc): 98.00 (93.00)
Generation # 440. Train Loss: 0.10. Train Acc (Test Acc): 97.00 (98.00)
Generation # 445. Train Loss: 0.12. Train Acc (Test Acc): 96.00 (94.80)
Generation # 450. Train Loss: 0.15. Train Acc (Test Acc): 96.00 (96.40)
Generation # 455. Train Loss: 0.09. Train Acc (Test Acc): 98.00 (95.00)
Generation # 460. Train Loss: 0.12. Train Acc (Test Acc): 96.00 (96.00)
Generation # 465. Train Loss: 0.10. Train Acc (Test Acc): 98.00 (95.20)
Generation # 470. Train Loss: 0.06. Train Acc (Test Acc): 97.00 (96.40)
Generation # 475. Train Loss: 0.20. Train Acc (Test Acc): 94.00 (97.40)
Generation # 480. Train Loss: 0.11. Train Acc (Test Acc): 97.00 (97.40)
Generation # 485. Train Loss: 0.12. Train Acc (Test Acc): 97.00 (97.40)
Generation # 490. Train Loss: 0.09. Train Acc (Test Acc): 98.00 (96.00)
Generation # 495. Train Loss: 0.11. Train Acc (Test Acc): 96.00 (96.80)
Generation # 500. Train Loss: 0.07. Train Acc (Test Acc): 98.00 (97.20)
[Finished in 270.1s]
'''
