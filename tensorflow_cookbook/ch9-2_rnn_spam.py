# coding: utf-8
# RNN


from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import os
import re
import io
import requests
from zipfile import ZipFile 

import numpy as np 
import matplotlib.pyplot as plt 

import tensorflow as tf 
from tensorflow.python.framework import ops 

ops.reset_default_graph()


# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs              = 20
batch_size          = 250
max_sequence_length = 25
rnn_size            = 10
embedding_size      = 50
min_work_frequency  = 10
learning_rate       = 0.0005
dropout_keep_prob   = tf.placeholder(tf.float32)


# Download or open data
data_dir  = ''
data_file = 'text_data.txt'
#if not os.path.exists(data_dir):
#	os.makedirs(data_dir)
print(os.path.isfile(os.path.join(data_dir, data_file)))

if not os.path.isfile(os.path.join(data_dir, data_file)):
	zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
	r = requests.get(zip_url)
	z = ZipFile(io.BytesIO(r.content))
	file = z.read('SMSSpamCollection')
	# Format Data
	text_data = file.decode('utf8')
	text_data = text_data.encode('ascii', errors='ignore')
	text_data = text_data.decode().split('\n')
	
	# Save data to text file
	with open(os.path.join(data_dir, data_file), 'w') as file_conn:
		for text in text_data:
			file_conn.write("{}\n".format(text))

else:
	# Open data from text file
	text_data = []
	with open(os.path.join(data_dir, data_file), 'r') as file_conn:
		for row in file_conn:
			text_data.append(row)
	#print(text_data[-3:])
	text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x)>=1]
#print(text_data[-3:])
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
#print(text_data_target[-3:], '\n', text_data_train[-3:])


'''
2018-03-29 11:44:49.616745: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
True
[Finished in 1.6s]
'''

'''
2018-03-29 11:47:38.590288: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
False

ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham	U dun say so early hor... U c already then say...
ham	Nah I don't think he goes to usf, he lives around here though
spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
...

[u'ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...', u'ham\tOk lar... Joking wif u oni...', u"spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", u'ham\tU dun say so early hor... U c already then say...', u"ham\tNah I don't think he goes to usf, he lives around here though", u"spam\tFreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok!
....
Only 10p per minute. BT-national-rate.', u'ham\tWill  b going to esplanade fr home?', u'ham\tPity, * was in mood for that. So...any other suggestions?', u"ham\tThe guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free", u'ham\tRofl. Its true to its name', u'']
[Finished in 3.7s]
'''

'''
2018-03-29 11:53:14.509997: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
True
["ham\tThe guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n", 'ham\tRofl. Its true to its name\n', '\n']
[Finished in 1.2s]

2018-03-29 11:54:15.121455: 
True
[['ham', 'Pity, * was in mood for that. So...any other suggestions?\n'], ['ham', "The guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n"], ['ham', 'Rofl. Its true to its name\n']]
[Finished in 1.5s]

2018-03-29 11:55:47.158290: 
True
['ham', 'ham', 'ham'] 
['Pity, * was in mood for that. So...any other suggestions?\n', "The guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n", 'Rofl. Its true to its name\n']
[Finished in 1.5s]
'''




# Create a text clearning function
def clean_text(text_string):
	text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
	text_string = " ".join(text_string.split())
	text_string = text_string.lower()
	return(text_string)

# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

'''
>>> import re
>>> s1 = "The guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n"
>>> s2 = 'Rofl. Its true to its name\n'
>>> 
>>> d = r'([^\s\w]|_|[0-9])+'
>>> d
'([^\\s\\w]|_|[0-9])+'
>>> 
>>> re.sub(d, '', s1)
'The guy did some bitching but I acted like id be interested in buying something else next week and he gave it to us for free\n'
>>> re.sub(d, '', s2)
'Rofl Its true to its name\n'
>>> t1 = re.sub(d, '', s1)
>>> t2 = re.sub(d, '', s2)
>>> " ".join(t1.split())
'The guy did some bitching but I acted like id be interested in buying something else next week and he gave it to us for free'
>>> " ".join(t2.split())
'Rofl Its true to its name'
>>> t1 = " ".join(t1.split())
>>> t2 = " ".join(t2.split())
>>> t1.lower()
'the guy did some bitching but i acted like id be interested in buying something else next week and he gave it to us for free'
>>> t2.lower()
'rofl its true to its name'
>>> 
>>> " ".join("a    b  e".split())
'a b e'
>>> "TAGEgae".lower()
'tagegae'
>>> 
'''


# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_work_frequency)
text_processed  = np.array(list(vocab_processor.fit_transform(text_data_train)))

'''
print("vocab_processor \t", vocab_processor)
print("vocab_processor.fit_transform() \t", vocab_processor.fit_transform(text_data_train))
print("text_processed 				   \t", text_processed.shape, text_processed )


2018-03-29 12:04:13.175027: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
True
vocab_processor 	 <tensorflow.contrib.learn.python.learn.preprocessing.text.VocabularyProcessor object at 0x7f37f7ab7590>
[Finished in 10.2s]

2018-03-29 12:06:01.150786: 
True
vocab_processor 		 <tensorflow.contrib.learn.python.learn.preprocessing.text.VocabularyProcessor object at 0x7fa2e3d305d0>
vocab_processor.fit_transform() 	 <generator object transform at 0x7fa2c3fba0f0>
text_processed 				   	 
[[ 44 455   0 ...,   0   0   0]
 [ 47 315   0 ...,   0   0   0]
 [ 46 465   9 ...,   0 368   0]
 ..., 
 [  0  59   9 ...,   0   0   0]
 [  5 493 108 ...,   1 198  12]
 [  0  40 474 ...,   0   0   0]]
[Finished in 2.4s]

2018-03-29 12:06:48.491224: 
True
vocab_processor 	 <tensorflow.contrib.learn.python.learn.preprocessing.text.VocabularyProcessor object at 0x7f4425f185d0>
vocab_processor.fit_transform() 	 <generator object transform at 0x7f44121a10f0>
text_processed 				   	 (5574, 25) 
[Finished in 2.3s]

'''



# Shuffle and split data
text_processed   = np.array(text_processed)
text_data_target = np.array([ 1 if x=='ham' else 0  for x in text_data_target])
shuffled_ix = np.random.permutation( np.arange(len(text_data_target)) )
x_shuffled  = text_processed[  shuffled_ix]
y_shuffled  = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[: ix_cutoff], x_shuffled[ix_cutoff :]
y_train, y_test = y_shuffled[: ix_cutoff], y_shuffled[ix_cutoff :]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size       : {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test) ))

'''
print("vocab_processor.vocabulary_ \t", vocab_processor.vocabulary_ )
print("vocab_size \t", vocab_size)


2018-03-29 12:12:35.971247: 
True
vocab_processor.vocabulary_ 	 <tensorflow.contrib.learn.python.learn.preprocessing.categorical_vocabulary.CategoricalVocabulary object at 0x7f662e5ecd10>
vocab_size 	 933
[Finished in 2.3s]

2018-03-29 12:14:02.044873: 
True
Vocabulary Size: 933
80-20 Train Test split: 4459 -- 1115
[Finished in 2.3s]
'''




# Create placeholders
x_data   = tf.placeholder(tf.int32, [None, max_sequence_length]) 							#[None, max_sequence_length], [None,25]
y_output = tf.placeholder(tf.int32, [None]) 												#[None, ]

# Create embedding
embedding_mat    = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)) 	#[vocab_size, embedding_size], [933,50]
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data) 		   #[None,max_sequence_length,embedding_size], [None,25,50]
#embedding_output_expanded = tf.expand_dims(embedding_output, -1) 	   #[None,max_sequence_length,embedding_size,1], [None,25,50,1]

'''
>>> import tensorflow as tf
>>> a = tf.constant([3,2])
>>> b = tf.expand_dims(a, -1)
>>> c = tf.expand_dims(b, -1)
>>> a
<tf.Tensor 'Const:0' shape=(2,) dtype=int32>
>>> b
<tf.Tensor 'ExpandDims:0' shape=(2, 1) dtype=int32>
>>> c
<tf.Tensor 'ExpandDims_3:0' shape=(2, 1, 1) dtype=int32>
>>> s = tf.Session()
2018-03-29 12:17:48.585512: 
>>> s.run(a)
array([3, 2], dtype=int32)
>>> s.run(b)
array([[3],
       [2]], dtype=int32)
>>> s.run(c)
array([[[3]],
       [[2]]], dtype=int32)
>>> 
'''




# Define the RNN cell
#tensorflow change >= 1.0, rnn is put into tensorflow.contrib.directory. Prior version not test.
if tf.__version__[0] >= '1':
	cell = tf.contrib.rnn.BasicRNNCell(num_units = rnn_size) 	#num_units=10
else:
	cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32) 
#													output: 		[None,max_sequence_length,rnn_size], [?,25,10]
#													last_states:	[None, rnn_size], [?,10]
output = tf.nn.dropout(output, dropout_keep_prob) 	# 				[None,max_sequence_length,rnn_size], [?,25,10]


# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2]) 						  # [max_sequence_length,None,rnn_size], [25,?,10]
last   = tf.gather(output, int(output.get_shape()[0]) - 1) 		  # [None, rnn_size], [?,10]


weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1)) 	#[rnn_size,2], [10,2]
bias   = tf.Variable(tf.constant(0.1, shape=[2])) 						#[2,]
logits_out = tf.matmul(last, weight) + bias 							#[None,2]


# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32, labels=int32
#									#[None,]
loss   = tf.reduce_mean(losses) 	#a scalar
'''
logits_out :	[None,2]
y_output   :	[None,]
	labels_y :		[None,2]
	
	predicts=tf.nn.softmax(logits=logits_out, dim=-1) :	[None,2]
	labels  =tf.clip_by_value(labels_y, 1e-10, 1.0)   :	[None,2]
	predicts=tf.clip_by_value(predicts, 1e-10, 1.0)   :	[None,2]
	cross_entropy = tf.reduce_sum(labels * tf.log(labels/predicts), axis=1)
		labels / predicts 	:	[None,2]
		tf.log(.) 			:	[None,2]
		labels*. 			:	[None,2]
		tf.reduce_sum(., 1) :	[None,]
'''


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32)) #scalar, tf.float32
'''
logits_out 	:	[None,2], tf.float32
y_output 	:	[None,] , tf.int32

	tf.argmax(logits_out, axis=1) 		:	[None,], tf.int64
	tf.cast(y_output, dtype=tf.int64) 	:	[None,], tf.int64
		tf.equal(., .) 						: 	[None,], tf.bool
			tf.cast(., dtype=tf.float32) 	:		[None,], tf.float32
				tf.reduce_mean(.) 			: 			a scalar, tf.float32
'''


optimizer  = tf.train.RMSPropOptimizer(learning_rate) 	#learning_rate=0.0005
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss  = []
train_accuracy = []
test_accuracy  = []
# Start training
for epoch in range(epochs): #20
	
	# Shuffle training data
	shuffled_ix = np.random.permutation(np.arange(len(x_train)))
	x_train = x_train[shuffled_ix]
	y_train = y_train[shuffled_ix]
	'''
	x_trn: (4459, 25)
	x_tst: (1115, 25)
	'''
	num_batches = int(len(x_train) / batch_size) + 1 
	'''
	batch_size = 250
	int(4459/250.)+1 = int(17.836)+1 = 17+1 = 18
	'''

	# TO DO Calculate Generations ExACTLY (exactly)
	for i in range(num_batches):
		# Select train data
		min_ix = i * batch_size
		max_ix = np.min([len(x_train), ((i+1) * batch_size)])
		x_train_batch = x_train[min_ix : max_ix]
		y_train_batch = y_train[min_ix : max_ix]

		# Run train step
		train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5}
		sess.run(train_step, feed_dict=train_dict)

	# Run loss and accuracy for training
	temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
	train_loss.append(temp_train_loss)
	train_accuracy.append(temp_train_acc)

	# Run Eval Step
	test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob: 1.0}
	temp_test_loss , temp_test_acc  = sess.run([loss, accuracy], feed_dict=test_dict )
	test_loss.append( temp_test_loss )
	test_accuracy.append( temp_test_acc )

	#print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
	print("Epoch {}: Train loss {:.4}, acc {:.4}. Test loss {:.4}, acc {:.4}.".format(epoch+1, temp_train_loss, temp_train_acc, temp_test_loss, temp_test_acc))


plt.figure(figsize=(12, 5))
plt.subplot(121)

# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss , 'r-' , label='Test Set' )
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
#plt.legend(loc='upper left')
#plt.show()

plt.legend(loc='upper right')
plt.subplot(122)

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy , 'r-' , label='Test Set' )
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.legend(loc='upper left')
#plt.show()

plt.legend(loc='lower right')
exe_this_time = 3 #2 #1 	#this_time
plt.savefig("ch9-2_fig{}.png".format(exe_this_time))




'''
2018-03-29 18:18:04.308847: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
True
Vocabulary Size       : 933
80-20 Train Test split: 4459 -- 1115
/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gradients_impl.py:96: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Epoch  1: Train loss 0.6608, acc 0.6794. Test loss 0.6491, acc 0.8251.
Epoch  2: Train loss 0.6302, acc 0.7464. Test loss 0.623 , acc 0.8296.
Epoch  3: Train loss 0.597 , acc 0.7703. Test loss 0.5874, acc 0.8296.
Epoch  4: Train loss 0.5596, acc 0.823 . Test loss 0.5465, acc 0.8287.
Epoch  5: Train loss 0.4899, acc 0.8565. Test loss 0.5055, acc 0.8305.
Epoch  6: Train loss 0.4655, acc 0.8421. Test loss 0.4724, acc 0.8359.
Epoch  7: Train loss 0.4631, acc 0.8421. Test loss 0.4496, acc 0.8368.
Epoch  8: Train loss 0.4876, acc 0.8182. Test loss 0.4328, acc 0.8404.
Epoch  9: Train loss 0.4572, acc 0.8278. Test loss 0.4208, acc 0.8457.
Epoch 10: Train loss 0.3484, acc 0.89  . Test loss 0.4124, acc 0.8484.
Epoch 11: Train loss 0.4111, acc 0.8421. Test loss 0.4072, acc 0.8511.
Epoch 12: Train loss 0.3829, acc 0.8708. Test loss 0.403 , acc 0.8565.
Epoch 13: Train loss 0.4549, acc 0.8325. Test loss 0.3995, acc 0.8592.
Epoch 14: Train loss 0.4826, acc 0.8469. Test loss 0.396 , acc 0.861 .
Epoch 15: Train loss 0.4125, acc 0.8469. Test loss 0.3927, acc 0.8673.
Epoch 16: Train loss 0.3205, acc 0.9091. Test loss 0.3895, acc 0.8682.
Epoch 17: Train loss 0.433 , acc 0.866 . Test loss 0.386 , acc 0.87  .
Epoch 18: Train loss 0.4096, acc 0.8517. Test loss 0.3824, acc 0.8709.
Epoch 19: Train loss 0.4637, acc 0.8278. Test loss 0.3778, acc 0.8726.
Epoch 20: Train loss 0.4111, acc 0.866 . Test loss 0.3716, acc 0.8735.
[Finished in 24.5s]


2018-03-29 18:23:45.485685: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
True
Vocabulary Size       : 933
80-20 Train Test split: 4459 -- 1115
/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gradients_impl.py:96: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Epoch 1 : Train loss 0.7119, acc 0.4067. Test loss 0.7051, acc 0.1695.
Epoch 2 : Train loss 0.6938, acc 0.555 . Test loss 0.6736, acc 0.8332.
Epoch 3 : Train loss 0.6376, acc 0.6986. Test loss 0.6282, acc 0.8377.
Epoch 4 : Train loss 0.5958, acc 0.7703. Test loss 0.5709, acc 0.843 .
Epoch 5 : Train loss 0.4946, acc 0.8612. Test loss 0.5133, acc 0.843.
Epoch 6 : Train loss 0.5176, acc 0.823 . Test loss 0.4696, acc 0.8439.
Epoch 7 : Train loss 0.4622, acc 0.8565. Test loss 0.4354, acc 0.843 .
Epoch 8 : Train loss 0.3973, acc 0.8756. Test loss 0.414 , acc 0.8448.
Epoch 9 : Train loss 0.3803, acc 0.8565. Test loss 0.4014, acc 0.8466.
Epoch 10: Train loss 0.4795, acc 0.8038. Test loss 0.3956, acc 0.8529.
Epoch 11: Train loss 0.4007, acc 0.8469. Test loss 0.3913, acc 0.8547.
Epoch 12: Train loss 0.4475, acc 0.8134. Test loss 0.3884, acc 0.8592.
Epoch 13: Train loss 0.4161, acc 0.8421. Test loss 0.3858, acc 0.8619.
Epoch 14: Train loss 0.3854, acc 0.8612. Test loss 0.3834, acc 0.8646.
Epoch 15: Train loss 0.3723, acc 0.8756. Test loss 0.3814, acc 0.8691.
Epoch 16: Train loss 0.3604, acc 0.8852. Test loss 0.3787, acc 0.8717.
Epoch 17: Train loss 0.4295, acc 0.8708. Test loss 0.3764, acc 0.8726.
Epoch 18: Train loss 0.4084, acc 0.866 . Test loss 0.3742, acc 0.8735.
Epoch 19: Train loss 0.3791, acc 0.8612. Test loss 0.3707, acc 0.8735.
Epoch 20: Train loss 0.4245, acc 0.8325. Test loss 0.3667, acc 0.8735.
[Finished in 23.0s]


2018-03-29 18:30:49.012963: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
True
Vocabulary Size       : 933
80-20 Train Test split: 4459 -- 1115
/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gradients_impl.py:96: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Epoch 1 : Train loss 0.622 , acc 0.7177. Test loss 0.5968, acc 0.8287.
Epoch 2 : Train loss 0.5942, acc 0.7703. Test loss 0.5763, acc 0.8314.
Epoch 3 : Train loss 0.5423, acc 0.8517. Test loss 0.5486, acc 0.8332.
Epoch 4 : Train loss 0.525 , acc 0.8325. Test loss 0.5162, acc 0.8314.
Epoch 5 : Train loss 0.504 , acc 0.8182. Test loss 0.4809, acc 0.8395.
Epoch 6 : Train loss 0.463 , acc 0.8421. Test loss 0.4524, acc 0.843 .
Epoch 7 : Train loss 0.464 , acc 0.823 . Test loss 0.4314, acc 0.8466.
Epoch 8 : Train loss 0.4329, acc 0.8517. Test loss 0.4176, acc 0.8493.
Epoch 9 : Train loss 0.4168, acc 0.8421. Test loss 0.4084, acc 0.8484.
Epoch 10: Train loss 0.3922, acc 0.8708. Test loss 0.4021, acc 0.8529.
Epoch 11: Train loss 0.4039, acc 0.8852. Test loss 0.397 , acc 0.8592.
Epoch 12: Train loss 0.3642, acc 0.866 . Test loss 0.3926, acc 0.8619.
Epoch 13: Train loss 0.4192, acc 0.8469. Test loss 0.3886, acc 0.8646.
Epoch 14: Train loss 0.4587, acc 0.8469. Test loss 0.3845, acc 0.8682.
Epoch 15: Train loss 0.3636, acc 0.8804. Test loss 0.3802, acc 0.8717.
Epoch 16: Train loss 0.4249, acc 0.8517. Test loss 0.3753, acc 0.8726.
Epoch 17: Train loss 0.3839, acc 0.866 . Test loss 0.3673, acc 0.8735.
Epoch 18: Train loss 0.4041, acc 0.8373. Test loss 0.3509, acc 0.8744.
Epoch 19: Train loss 0.3489, acc 0.8565. Test loss 0.3186, acc 0.8807.
Epoch 20: Train loss 0.3405, acc 0.89  . Test loss 0.2872, acc 0.8888.
[Finished in 22.7s]
'''



