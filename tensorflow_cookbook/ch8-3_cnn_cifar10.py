# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/08_Convolutional_Neural_Networks/03_CNN_CIFAR10/03_cnn_cifar10.py

from __future__ import division
from __future__ import print_function


# More Advanced CNN Model: CIFAR-10
#---------------------------------------
#
# In this example, we will download the CIFAR-10 images
# and build a CNN model with dropout and regularization
#
# CIFAR is composed ot 50k train and 10k test
# images that are 32x32.

import os
import sys
import tarfile
from six.moves import urllib

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops

ops.reset_default_graph()


# Change Directory
abspath = os.path.abspath(__file__)
dname   = os.path.dirname(abspath)
os.chdir(dname)

'''
print(abspath)
print(dname)

/home/ubuntu/Program/Code/tensorflow_cookbook/ch8-2_cnn_cifar10.py
[Finished in 17.2s]

/home/ubuntu/Program/Code/tensorflow_cookbook/ch8-2_cnn_cifar10.py
/home/ubuntu/Program/Code/tensorflow_cookbook
[Finished in 1.6s]
'''


# Start a graph session
sess = tf.Session()

# Set model parameters
batch_size   = 128
data_dir     = ''   ##'temp'
output_every = 100  ##50
generations  = 2000 ##20000
eval_every   = 50   ##500
image_height = 32
image_width  = 32
crop_height  = 24
crop_width   = 24
num_channels = 3
num_targets  = 10
extract_folder = 'cifar-10-batches-bin'

# Exponential Learning Rate Decay Params
learning_rate = 0.1
lr_decay      = 0.1
num_gens_to_wait = 250.

# Extract model parameters
image_vec_length = image_height * image_width * num_channels
record_length    = 1 + image_vec_length # ( + 1 for the 0-9 label)


# Load data
##data_dir = 'temp'
##if not os.path.exists(data_dir):
##	os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Check if file exists, otherwise download it
data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if os.path.isfile(data_file):
	pass
else:
	# Download file
	def progress(block_num, block_size, total_size):
		progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
		print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
	filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
	# Extract file
	tarfile.open(filepath, 'r:gz').extractall(data_dir)

'''
2018-03-02 23:27:21.905621: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX

 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 0.00%
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 0.01%
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 6.66% 
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 66.70%
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 99.97%
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 99.97%
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 99.99%
 Downloading http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz - 100.00%[Finished in 192.5s]

 2018-03-02 23:38:22.384742: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[Finished in 1.6s]
'''


# Define CIFAR reader
def read_cifar_files(filename_queue, distort_images=True):
	reader = tf.FixedLengthRecordReader(record_bytes=record_length)
	key, record_string = reader.read(filename_queue)
	record_bytes = tf.decode_raw(record_string, tf.uint8)
	image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)

	# Extract image
	image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
								 [num_channels, image_height, image_width])

	# Reshape image
	image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
	reshaped_image   = tf.cast(image_uint8image, tf.float32)
	# Randomly Crop image
	final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)

	if distort_images:
		# Randomly flip the image horizontally, change the brightness and contrast
		final_image = tf.image.random_flip_left_right(final_image)
		final_image = tf.image.random_brightness(final_image, max_delta=63)
		final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)

	# Normalize whitening
	final_image = tf.image.per_image_standardization(final_image)
	return(final_image, image_label)

'''
>>> tf.FixedLengthRecordReader(record_bytes=record_length)
	<tensorflow.python.ops.io_ops.FixedLengthRecordReader object at 0x7f2d46fecb90>
>>> reader = tf.FixedLengthRecordReader(record_bytes=record_length)
>>> reader
	<tensorflow.python.ops.io_ops.FixedLengthRecordReader object at 0x7f2d46fec890>
>>> ftrn
	['temp/cifar-10-batches-bin/data_batch_1.bin', 'temp/cifar-10-batches-bin/data_batch_2.bin', 'temp/cifar-10-batches-bin/data_batch_3.bin', 'temp/cifar-10-batches-bin/data_batch_4.bin', 'temp/cifar-10-batches-bin/data_batch_5.bin']
>>> ftst
	['temp/cifar-10-batches-bin/test_batch.bin']
>>> fqtrn = tf.train.string_input_producer(ftrn)
>>> fqtst = tf.train.string_input_producer(ftst)
>>> fqtrn
	<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7f2d46f680d0>
>>> fqtst
	<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7f2d46f7b610>
>>> key1, record_string1 = reader.read(fqtst)
>>> print "len(key1)", len(key1), "len(record_string1)", len(record_string1)
len(key1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'Tensor' has no len()
>>> key1
	<tf.Tensor 'ReaderReadV2:0' shape=() dtype=string>
>>> record_string1
	<tf.Tensor 'ReaderReadV2:1' shape=() dtype=string>
>>> key2, record_string2 = reader.read(fqtrn)
>>> key2
	<tf.Tensor 'ReaderReadV2_1:0' shape=() dtype=string>
>>> record_string2
	<tf.Tensor 'ReaderReadV2_1:1' shape=() dtype=string>
>>> record_bytes1 = tf.decode_raw(record_string1, tf.uint8)
>>> record_bytes2 = tf.decode_raw(record_string2, tf.uint8)
>>> record_bytes1
	<tf.Tensor 'DecodeRaw:0' shape=(?,) dtype=uint8>
>>> record_bytes2
	<tf.Tensor 'DecodeRaw_1:0' shape=(?,) dtype=uint8>
>>> 


>>> data_dir = ""
>>> extract_folder = "cifar-10-batches-bin"
>>> import os.path
>>> ftrn = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
>>> ftst = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
>>> import tensorflow as tf
>>> fqtrn = tf.train.string_input_producer(ftrn)
>>> fqtst = tf.train.string_input_producer(ftst)
>>> record_length = 1+32*32*3
>>> record_length
3073
>>> frtrn = tf.FixedLengthRecordReader(record_bytes=record_length)
>>> frtst = tf.FixedLengthRecordReader(record_bytes=record_length)
>>> ftrn
['cifar-10-batches-bin/data_batch_1.bin', 'cifar-10-batches-bin/data_batch_2.bin', 'cifar-10-batches-bin/data_batch_3.bin', 'cifar-10-batches-bin/data_batch_4.bin', 'cifar-10-batches-bin/data_batch_5.bin']
>>> ftst
['cifar-10-batches-bin/test_batch.bin']
>>> fqtrn
<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7fdc68e99550>
>>> fqtst
<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7fdc5ab91590>
>>> frtrn
<tensorflow.python.ops.io_ops.FixedLengthRecordReader object at 0x7fdc77a84210>
>>> frtst
<tensorflow.python.ops.io_ops.FixedLengthRecordReader object at 0x7fdc77a89710>
>>> 
>>> key1, recstr1 = frtrn.read(fqtrn)
>>> key2, recstr2 = frtst.read(fqtst)
>>> recb1 = tf.decode_raw(recstr1, tf.uint8)
>>> recb2 = tf.decode_raw(recstr2, tf.uint8)
>>> key1
<tf.Tensor 'ReaderReadV2:0' shape=() dtype=string>
>>> key2
<tf.Tensor 'ReaderReadV2_1:0' shape=() dtype=string>
>>> recstr1
<tf.Tensor 'ReaderReadV2:1' shape=() dtype=string>
>>> recstr2
<tf.Tensor 'ReaderReadV2_1:1' shape=() dtype=string>
>>> recb1
<tf.Tensor 'DecodeRaw:0' shape=(?,) dtype=uint8>
>>> recb2
<tf.Tensor 'DecodeRaw_1:0' shape=(?,) dtype=uint8>
>>> 
>>> tf.slice(recb1, [0], [1])
<tf.Tensor 'Slice:0' shape=(1,) dtype=uint8>
>>> tf.slice(recb2, [0], [1])
<tf.Tensor 'Slice_1:0' shape=(1,) dtype=uint8>
>>> imglb1 = tf.cast(tf.slice(recb1, [0], [1]), tf.int32)
>>> imglb2 = tf.cast(tf.slice(recb2, [0], [1]), tf.int32)
>>> s = tf.Session()
2018-03-03 09:27:50.527707: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
>>> 



>>> data_dir = ""
>>> extract_folder = "cifar-10-batches-bin"
>>> import os.path
>>> import tensorflow as tf
>>> ftst = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
>>> fqtst = tf.train.string_input_producer(ftst)
>>> record_length = 3073
>>> frtst = tf.FixedLengthRecordReader(record_bytes=record_length)
>>> key2, recstr2 = frtst.read(fqtst)
>>> recb2 = tf.decode_raw(recstr2, tf.uint8)
>>> s = tf.Session()
	2018-03-03 09:32:12.549204: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
>>> tem2 = tf.slice(recb2, [0], [1])
>>> tem2
	<tf.Tensor 'Slice:0' shape=(1,) dtype=uint8>
>>> tem2[0]
	<tf.Tensor 'strided_slice:0' shape=() dtype=uint8>
>>> imglb2 = tf.cast(tf.slice(recb2, [0], [1]), tf.int32)
>>> imglb2
	<tf.Tensor 'Cast:0' shape=(1,) dtype=int32>
>>> imglb2[0]
	<tf.Tensor 'strided_slice_2:0' shape=() dtype=int32>
>>> 
>>> tf.slice(recb2, [1], [record_length-1])
<tf.Tensor 'Slice_2:0' shape=(3072,) dtype=uint8>
>>> tf.slice(recb2, [1], [record_length-1])[0]
<tf.Tensor 'strided_slice_3:0' shape=() dtype=uint8>
>>> tem2 = tf.slice(recb2, [1], [record_length-1])
>>> imgex2 = tf.reshape(tem2, [3,32,32])
>>> imgex2
<tf.Tensor 'Reshape:0' shape=(3, 32, 32) dtype=uint8>
>>> imgui2 = tf.transpose(imgex2, [1,2,0])
>>> imgui2
<tf.Tensor 'transpose:0' shape=(32, 32, 3) dtype=uint8>
>>> imgre2 = tf.cast(imgui2, tf.float32)
>>> imgre2
<tf.Tensor 'Cast_1:0' shape=(32, 32, 3) dtype=float32>
>>> imgfn2 = tf.image.resize_image_with_crop_or_pad(imgre2, 24,24)
>>> imgfn2
<tf.Tensor 'Squeeze:0' shape=(24, 24, 3) dtype=float32>
>>> 
>>> tf.image.random_flip_left_right(imgfn2)
<tf.Tensor 'cond/Merge:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn2 = tf.image.random_flip_left_right(imgfn2)
>>> tf.image.random_brightness(imgfn2, max_delta=63)
<tf.Tensor 'adjust_brightness/Identity_1:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn2 = tf.image.random_brightness(imgfn2, max_delta=63)
>>> imgfn2
<tf.Tensor 'adjust_brightness_1/Identity_1:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn2 = tf.image.random_contrast(imgfn2, lower=0.2, upper=1.8)
>>> imgfn2
<tf.Tensor 'adjust_contrast/Identity_1:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn2
<tf.Tensor 'adjust_contrast/Identity_1:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn2 = tf.image.per_image_standardization(imgfn2)
>>> imgfn
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'imgfn' is not defined
>>> imgfn2
<tf.Tensor 'div:0' shape=(24, 24, 3) dtype=float32>
>>> imglb2
<tf.Tensor 'Cast:0' shape=(1,) dtype=int32>
>>> 

'''



# Create a CIFAR image pipeline from reader
def input_pipeline(batch_size, train_logical=True):
	if train_logical:
		files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
	else:
		files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
	filename_queue = tf.train.string_input_producer(files)
	image, label = read_cifar_files(filename_queue)

	# min_after_dequeue defines how big a buffer we will randomly sample
	#   from -- bigger means better shuffling but slower start up and more
	#   memory used.
	# capacity must be larger than min_after_dequeue and the amount larger
	#   determines the maximum we will prefetch.  Recommendation:
	#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
	min_after_dequeue = 5000
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch([image, label],
								 batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return(example_batch, label_batch)

'''
>>> data_dir = ''
>>> extract_folder = 'cifar-10-batches-bin'
>>> import os
>>> import os.path
>>> os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(3))
'cifar-10-batches-bin/data_batch_3.bin'
>>> range(1,6)
[1, 2, 3, 4, 5]
>>> data_dir = 'temp'
>>> os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(3))
'temp/cifar-10-batches-bin/data_batch_3.bin'
>>> 
>>> os.path.join(data_dir, extract_folder, 'test_batch.bin')
	'temp/cifar-10-batches-bin/test_batch.bin'
>>> [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i))  for i in range(1,6)]
	['temp/cifar-10-batches-bin/data_batch_1.bin', 'temp/cifar-10-batches-bin/data_batch_2.bin', 'temp/cifar-10-batches-bin/data_batch_3.bin', 'temp/cifar-10-batches-bin/data_batch_4.bin', 'temp/cifar-10-batches-bin/data_batch_5.bin']
>>> 

>>> ftst = os.path.join(data_dir, extract_folder, 'test_batch.bin')
>>> ftrn = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i))  for i in range(1,6)]
>>> tf.train.string_input_producer(ftst)
	Traceback (most recent call last):
	ValueError: Shape () must have rank at least 1
>>> tf.train.string_input_producer(ftrn)
<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7f2d46fabc50>
>>> ftst 
'temp/cifar-10-batches-bin/test_batch.bin'
>>> ftst  = [ftst]
>>> tf.train.string_input_producer(ftst)
<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7f2d46fd5390>
>>> 



>>> import os.path
>>> import tensorflow as tf
>>> f = [os.path.join("", "cifar-10-batches-bin", 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
>>> f
	['cifar-10-batches-bin/data_batch_1.bin', 'cifar-10-batches-bin/data_batch_2.bin', 'cifar-10-batches-bin/data_batch_3.bin', 'cifar-10-batches-bin/data_batch_4.bin', 'cifar-10-batches-bin/data_batch_5.bin']
>>> fq = tf.train.string_input_producer(f)
>>> fr = tf.FixedLengthRecordReader(record_bytes=3073)
>>> key,recstr = fr.read(fq)
>>> fq
		<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7fb2f5258510>
>>> fr 
		<tensorflow.python.ops.io_ops.FixedLengthRecordReader object at 0x7fb303e621d0>
>>> key 														<tf.Tensor 'ReaderReadV2:0' shape=() dtype=string>
>>> recstr 														<tf.Tensor 'ReaderReadV2:1' shape=() dtype=string>
>>> recb = tf.decode_raw(recstr, tf.uint8) 							<tf.Tensor 'DecodeRaw:0' shape=(?,) dtype=uint8>
>>> tf.slice(recb, [0], [1]) 										<tf.Tensor 'Slice:0' shape=(1,) dtype=uint8>
>>> imglb = tf.cast(tf.slice(recb, [0], [1]), tf.int32) 			<tf.Tensor 'Cast:0' shape=(1,) dtype=int32>
>>> tf.slice(recb, [1], [3072]) 									<tf.Tensor 'Slice_2:0' shape=(3072,) dtype=uint8>
>>> imgex = tf.reshape(tf.slice(recb,[1],[3072]) ,[3,32,32]) 		<tf.Tensor 'Reshape:0' shape=(3, 32, 32) dtype=uint8>
>>> imgui = tf.transpose(imgex, [1,2,0]) 							<tf.Tensor 'transpose:0' shape=(32, 32, 3) dtype=uint8>
>>> imgre = tf.cast(imgui, tf.float32) 								<tf.Tensor 'Cast_1:0' shape=(32, 32, 3) dtype=float32>
>>> imgfn = tf.image.resize_image_with_crop_or_pad(imgre, 24,24) 	<tf.Tensor 'Squeeze:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn = tf.image.random_flip_left_right(imgfn) 					<tf.Tensor 'cond/Merge:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn = tf.image.random_brightness(imgfn, max_delta=63) 	
													<tf.Tensor 'adjust_brightness/Identity_1:0' shape=(24, 24, 3) dtype=float32>
>>> imgfn = tf.image.random_contrast(imgfn, lower=0.2, upper=1.8) 	
													<tf.Tensor 'adjust_contrast/Identity_1:0' shape=(24,24,3) dtype=float32>
>>> imgfn = tf.image.per_image_standardization(imgfn)
>>> imgfn
	<tf.Tensor 'div:0' shape=(24, 24, 3) dtype=float32>
>>> imglb
	<tf.Tensor 'Cast:0' shape=(1,) dtype=int32>
>>> 
>>> batch_size= 128
>>> min_af_dq = 5000
>>> cap = min_af_dq + 3*batch_size
>>> cap
		5384
>>> eg_batch, lb_batch = tf.train.shuffle_batch([imgfn, imglb], batch_size=batch_size, capacity=cap, min_after_dequeue = min_af_dq)
>>> eg_batch
	<tf.Tensor 'shuffle_batch:0' shape=(128, 24, 24, 3) dtype=float32>
>>> lb_batch
	<tf.Tensor 'shuffle_batch:1' shape=(128, 1) dtype=int32>
>>> 

'''



# Get data
print('Getting/Transforming Data.')
# Initialize the data pipeline
images, targets = input_pipeline(batch_size, train_logical=True)
# Get batch test images and targets from pipeline
test_images, test_targets = input_pipeline(batch_size, train_logical=True)

print("train - images\t", images )
print("train - labels\t", targets)
print("test  - images\t", images )
print("test  - labels\t", targets)

'''
2018-03-03 08:59:56.238275: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Getting/Transforming Data.
train - images Tensor("shuffle_batch:0", shape=(128, 24, 24, 3), dtype=float32)
train - labels Tensor("shuffle_batch:1", shape=(128, 1), dtype=int32)
test  - images Tensor("shuffle_batch:0", shape=(128, 24, 24, 3), dtype=float32)
test  - labels Tensor("shuffle_batch:1", shape=(128, 1), dtype=int32)
[Finished in 12.7s]

2018-03-03 09:01:15.585269: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Getting/Transforming Data.
train - images	 Tensor("shuffle_batch:0", shape=(128, 24, 24, 3), dtype=float32)
train - labels	 Tensor("shuffle_batch:1", shape=(128, 1), dtype=int32)
test  - images	 Tensor("shuffle_batch:0", shape=(128, 24, 24, 3), dtype=float32)
test  - labels	 Tensor("shuffle_batch:1", shape=(128, 1), dtype=int32)
[Finished in 1.7s]
'''






# Define the model architecture, this will return logits from images
def cifar_cnn_model(input_images, batch_size, train_logical=True):
	def truncated_normal_var(name, shape, dtype):
		return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
	def zero_var(name, shape, dtype):
		return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

	# First Convolutional Layer
	with tf.variable_scope('conv1') as scope:
		# Conv_kernel is 5x5 for all 3 colors and we will create 64 features
		conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5,5,3,64], dtype=tf.float32)
		# We convolve across the image with a stride size of 1
		conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1,1,1,1], padding='SAME')
		# Initialize and add the bias term
		conv1_bias   = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
		conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
		# ReLU element wise
		relu_conv1 = tf.nn.relu(conv1_add_bias)
		'''
		conv1 		#[]
			input_images 	#[batch_size,24,24,3]
			conv1_kernel 	#[5,5,3,64]
			strides 		#[1,1,1,1]
				(1) filter  ->  [5*5*3, 64]  ->  [75,64]
				(2) patch   ->  [batch_size, out_h1,out_w1, 5*5*3]  ->  [batch_size, out_h1,out_w1, 75]
				(3) 	output: [batch_size, out_h1, out_w1, 64]
				(i ) 	out_h1  =  ceil(24/ strides[1] ) = ceil(24/1) = 24
				(ii) 	out_w1  =  ceil(24/ strides[2] ) = ceil(24/1) = 24

		relu_conv1 	#[]
			conv1 			#[batch_size, out_h1, out_w1, 64]
			conv1_bias 		#[64,]
				(4) conv1_add_bias 	->  [batch_size, out_h1, out_w1, 64]
				(5) relu_conv1 		->  [batch_size, out_h1, out_w1, 64]
				(iii) 	output:	[batch_size, out_h1, out_w1, 64]	->	[batch_size, 24,24,64]
		'''
	print('def cifar_cnn_model:')
	print('\t scope: conv1')
	print('\t\t', conv1_kernel.name, '\n\t\t', conv1_bias.name, '\n\t\t', conv1.name, conv1_add_bias.name, relu_conv1.name)

	# Max Pooling
	pool1 = tf.nn.max_pool(relu_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool_layer1')
	'''
		pool1 		#[]
			relu_conv1 		#[batch_size, out_h1, out_w1, 64]
			ksize 			#[1, 3, 3, 1]
			strides 		#[1, 2, 2, 1]
				(6) 	output: [batch_size, new_h1, new_w1, 64]
				(iv) 	new_h1 	= ceil(out_h/ strides[1] ) = ceil(24/2) = 12
				(v ) 	new_w1 	= ceil(out_w/ strides[2] ) = ceil(24/2) = 12
	'''

	# Local Response Normalization (parameters from paper)
	# paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
	norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
	'''
		norm1 		#[]
			pool1 			#[batch_size, new_h1, new_w1, 64]
				(7) 	output: same as 'pool1'
	'''
	print('\t\t * \t', pool1.name, norm1.name)

	# Second Convolutional Layer
	with tf.variable_scope('conv2') as scope:
		# Conv kernel is 5x5, across all prior 64 features and we create 64 more features
		conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5,5,64,64], dtype=tf.float32)
		# Convolve filter across prior output with stride size of 1
		conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1,1,1,1], padding='SAME')
		# Initialize and add the bias
		conv2_bias   = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
		conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
		# ReLU element wise
		relu_conv2 = tf.nn.relu(conv2_add_bias)
		'''
		conv2 		#[]
			norm1 			#[batch_size, new_h1, new_w1,  64]
			conv2_kernel 	#[5, 5, 64, 64]
			strides 		#[1, 1, 1, 1]
				(1) filter 	->  [5*5*64, 64]  ->  [1600,64]
				(2) patch 	->  [batch_size, out_h2,out_w2, 5*5*64]  ->  [batch_size,out_h2,out_w2, 1600]
				(3) 	output: [batch_size, out_h2, out_w2, 64]
				(i ) 	out_h2  =  ceil(new_h1/ strides[1] ) = ceil(12/1) = 12
				(ii) 	out_w2  =  ceil(new_w1/ strides[2] ) = ceil(12/1) = 12

		relu_conv2 	#[]
			conv2 			#[batch_size, out_h2, out_w2, 64]
			conv2_bias 		#[64,]
				(4) conv2_add_bias 	=  [batch_size, out_h2, out_w2, 64]
				(5) relu_conv2 		=  [batch_size, out_h2, out_w2, 64]
				(iii) 	output:	[batch_size, out_h2, out_w2, 64] 	->  [batch_size, 12,12,64]
		'''
	print('\t scope: conv2')
	print('\t\t', conv2_kernel.name, '\n\t\t', conv2_bias.name, '\n\t\t', conv2.name, conv2_add_bias.name, relu_conv2.name)

	# Max Pooling
	pool2 = tf.nn.max_pool(relu_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool_layer2')
	'''
		pool2 		#[]
			relu_conv2 		#[batch_size, out_h2, out_w2, 64] 		->  [batch_size, 12,12,64]
			ksize 			#[1, 3, 3, 1]
			strides 		#[1, 2, 2, 1]
				(6) 	output: [batch_size, new_h2, new_w2, 64]
				(iv) 	new_h2  =  ceil(out_h2/ strides[1] ) = ceil(12/2) = 6
				(v ) 	new_w2  =  ceil(out_w2/ strides[2] ) = ceil(12/2) = 6
	'''

	# Local Response Normalization (parameters from paper)
	norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
	'''
		norm2 		#[]
			pool2 			#[batch_size, new_h2, new_w2, 64] 		->  [batch_size, 6, 6, 64]
				(7) 	output: same as 'pool2'
	'''
	print('\t\t * \t', pool2.name, norm2.name)

	# Reshape output into a single matrix for multiplication for the fully connected layers
	reshape_output = tf.reshape(norm2, [batch_size, -1])
	reshape_dim    = reshape_output.get_shape()[1].value  #or: .get_shape().as_list()[1]
	'''
		reshape_output 	#[]
			norm2 			#[batch_size, new_h2, new_w2, 64] 		->  [batch_size, 6, 6, 64]
				(1) 	output: [batch_size, new_h2 * new_w2 * 64] 	->  [batch_size, 6*6*64]  ->  [batch_size,2304]
		reshape_dim     #a number
				(2) 	output: new_h2 * new_w2 * 64 				->  6*6*64 = 2304
	'''
	print('\t reshape \t', reshape_output.name, reshape_dim.name)

	# First Fully Connected Layer
	with tf.variable_scope('full1') as scope:
		# Fully connected layer will have 384 outputs.
		full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshape_dim, 384], dtype=tf.float32)
		full_bias1   = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
		full_layer1  = tf.nn.relu(tf.add(tf.matmul(reshape_output, full_weight1), full_bias1))
		'''
		full_layer1 	#[]
			reshape_output 	#[batch_size, reshape_dim] 				->  [batch_size,2304]
			full_weight1 	#[reshape_dim, 384]
			full_bias1 		#[384,]
				(1) tf.matmul(reshape_output, full_weight1) 	#[batch_size, 384]
				(2) tf.add(., full_bias1) 						#[batch_size, 384]
				(3) tf.nn.relu(.) 								#[batch_size, 384]
		'''
	print('\t scope: full1')
	print('\t\t ')

	# Second Fully Connected Layer
	with tf.variable_scope('full2') as scope:
		# Second fully connected layer has 192 outputs.
		full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
		full_bias2   = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
		full_layer2  = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
		'''
		full_layer2 	#[]
			full_layer1 	#[batch_size, 384]
			full_weight2 	#[384, 192]
			full_bias2 		#[192,]
				(1) tf.matmul(full_layer1, full_weight2) 		#[batch_size, 192]
				(2) tf.add(., full_bias2) 						#[batch_size, 192]
				(3) tf.nn.relu(.) 								#[batch_size, 192]
		'''

	# Final Fully Connected Layer -> 10 categories for output (num_targets)
	with tf.variable_scope('full3') as scope:
		# Final fully connected layer has 10 (num_targets) outputs.
		full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
		full_bias3   = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
		final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
		'''
		final_output 	#[]
			full_layer2 	#[batch_size, 192]
			full_weight3 	#[192, num_targets]
			full_bias3 		#[num_targets,]
				(1) tf.matmul(full_layer2, full_weight3) 		#[batch_size, num_targets]
				(2) tf.add(., full_bias3) 						#[batch_size, num_targets]
		'''

	return(final_output)  		#[batch_size, num_targets]



# Loss function
def cifar_loss(logits, targets):
	# Get rid of extra dimensions and cast targets into integers
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	# Calculate cross entropy from logits and targets
	cross_entropy = tf.nn.sqarse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
	# Take the average loss across batch size
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	return(cross_entropy_mean)
	'''
	Input:
		logits 		#[batch_size, num_targets]  ->  [batch_size,10], tf.float32
		*images 	#[batch_size, 24, 24, 3], tf.float32
		targets 	#[batch_size, 1]		, tf.int32 
	Output:
		cross_entropy_mean:	#a number

	Processing:
		targets 			#[]
			(1)	tf.cast(targets, .) 	#[batch_size, 1]
			(2) tf.squeeze(.) 			#[batch_size,]
		cross_entropy 		#[]
			(3) logits 			#[batch_size, num_targets]
							->   [batch_size,]
				targets 		#[batch_size,]
			(4) output: 				#[batch_size,]
		cross_entropy_mean 	#a number
	'''


# Train step
def train_step(loss_value, generation_num):
	# Our learning rate is an exponential decay after we wait a fair number of generations
	model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num, 
													 num_gens_to_wait, lr_decay, staircase=True)
	# Create optimizer
	my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
	# Initialize train step
	train_step = my_optimizer.minimize(loss_value)
	return(train_step)

	'''
	Input:
		loss_value 			#a number, =
		generation_num 		#a number, =
	Output:
		train_step

	Parameters:
		batch_size 			= 128 		output_every = 50 	eval_every = 500
		generations 		= 20000 -> 2000
		learning_rate 		= 0.1
		lr_decay 			= 0.1
		num_gens_to_wait 	= 250.

		tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
		decayed_learning_rate =    learning_rate * decay_rate ^ (global_step / decay_steps)

	Processing:
		model_learning_rate 	#[]
			learning_rate 	 = 0.1
			generation_num 	 = 20000
			num_gens_to_wait = 250.
			lr_decay 		 = 0.1
				(1) 		0.1*0.1^(20000/250.) = 0.1* 0.1^(80) = 
							>>> np.power(0.1, 80) 			1.0000000000000045e-80
							>>> np.power(0.1, 80)*0.1 		1.0000000000000045e-81

	Processing - updated:
		model_learning_rate 	#a number
			learning_rate 	 = 0.1
			generation_num 	 = 0 = 0.
			num_gens_to_wait = 250.
			lr_decay 		 = 0.1
				(1) 		0.1*0.1^(0/250) = 0.1* 0.1^0 = 0.1*1.0 = 0.1
	'''


# Accuracy function
def accuracy_of_batch(logits, targets):
	# Make sure targets are integers and drop extra dimensions
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	# Get predicted values by finding which logit is the greatest
	batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
	# Check if they are equal across the batch
	predicted_correctly = tf.equal(batch_predictions, targets)
	# Average the 1's and 0's (True's and False's) across the batch size
	accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
	return(accuracy)
	'''
	Input:
		logits 		#[batch_size, num_targets], tf.float32 	->  [batch_size,10]
		targets 	#[batch_size, 1] 		  , tf.int32
	Output:
		accuracy 	#a number

	Processing:
		targets 
			(1) tf.cast(., dtype=tf.int32) 		#[batch_size, 1]
			(2) tf.squeeze(.) 					#[batch_size,]
		batch_predictions 
			(3) tf.argmax(logits, axis=1) 		#[batch_size,], tf.int64?
			(4) tf.cast(., dtype=tf.int32) 		#[batch_size,]
		predicted_correctly
			(5) tf.equal(., targets) 			#[batch_size,], tf.bool
		accuracy 
			(6) tf.cast(., dtype=tf.float32) 	#[batch_size,], tf.float32
			(7) tf.reduce_mean(.) 				#a number
	'''



# Get data
##print('Getting/Transforming Data.')
# Initialize the data pipeline
##images, targets = input_pipeline(batch_size, train_logical=True)
# Get batch test images and targets from pipline
##test_images, test_targets = input_pipeline(batch_size, train_logical=False)


# Declare Model
print('Creating the CIFAR10 Model.')
with tf.variable_scope('model_definition') as scope:
	# Declare the training network model
	model_output = cifar_cnn_model(images, batch_size)
	# This is very important!!!  We must set the scope to REUSE the variables,
	#  otherwise, when we set the test network model, it will create new random
	#  variables.  Otherwise we get random evaluations on the test batches.
	scope.reuse_variables()
	test_output  = cifar_cnn_model(test_images, batch_size)


# Declare loss function
print('Declare Loss Function.')
loss = cifar_loss(model_output, targets)

# Create accuracy function
accuracy = accuracy_of_batch(test_output, test_targets)

# Create training operations
print('Creating the Training Operation.')
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# Initialize Variables
print('Initializing the Variables.')
init = tf.global_variables_initializer()
sess.run(init)

# Initialize queue (This queue will feed into the model, so no placeholders necessary)
tf.train.start_queue_runners(sess=sess)

# Train CIFAR Model
print('Starting Training')
train_loss = []
test_accuracy = []
for i in range(generations):
	_, loss_value = sess.run([train_op, loss])

	if (i+1) % output_every == 0:
		train_loss.append(loss_value)
		output = 'Generation {}: Loss = {:.5f}'.format((i+1), loss_value)
		print(output)

	if (i+1) % eval_every == 0:
		[temp_accuracy] = sess.run([accuracy])
		test_accuracy.append(temp_accuracy)
		acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
		print(acc_output)


# Print loss and accuracy
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)

plt.figure(figsize=(12, 5))
plt.subplot(121)
# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
#plt.show()

plt.subplot(122)
# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()






