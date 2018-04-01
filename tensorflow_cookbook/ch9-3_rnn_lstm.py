# coding: utf-8
# RNN - LSTM


from __future__ import division
from __future__ import print_function

import os
import re
import string
import requests
import collections
import random
import pickle

import numpy as np 

#import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import tensorflow as tf 
from tensorflow.python.framework import ops 

ops.reset_default_graph()
sess = tf.Session()


'''
Descriptions of parameters:
min_word_fred 		:	Only attempt to model words that appear at least 5 times
rnn_size 			:	Size of our RNN (equal to the embedding size)
epochs 				:	Number of epochs to cycle through the data
batch_size 			:	How many examples to train on at once
learning_rate 		:	The learning rate or the convergence parameter
training_seq_len 	:	The length of the surrounding word group (e.g. 10 = 5 on each side)
embedding_size 		:	Must be equal to the ``rnn_size''
save_every 			:	How often to save the model
eval_every 			:	How often to evaluate the model
prime_texts 		:	List of test sentences
'''

# Set RNN Parameters
min_word_fred 	 = 5 		# Trim the less frequent words off
rnn_size 		 = 128 		# RNN Model size
embedding_size 	 = 100 		# Word embedding size
epochs 			 = 10 		# Number of epochs to cycle through data
batch_size 		 = 100 		# Train on this many examples at once
learning_rate 	 = 0.001 	# Learning rate
training_seq_len = 50 		# How long of a word group to consider
embedding_size 	 = rnn_size
save_every 		 = 500 		# How often to save model checkpoints
eval_every 		 = 50 		# How often to evaluate the test sentences
prime_texts 	 = ['thou art more', 'to be or not to', 'wherefore art thou']

# Download/store Shakespeare data
data_dir   = '' 	#'temp'
data_file  = 'shakespeare.txt'
model_path = 'shakespeare_model'
full_model_dir = os.path.join(data_dir, model_path)

# Declare punctuation to remove, everything except hyphens and apostrophes
punctuation = string.punctuation
punctuation = ''.join([ x  for x in punctuation  if x not in ['-', "'"]])
'''
>>> string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
>>> 
>>> punct = string.punctuation
>>> ''.join( [x for x in punct  if x not in ['-',"'"]])
'!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
>>> ''.join( [x for x in punct])
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
'''


# Make Model Directory
if not os.path.exists(full_model_dir):
	os.makedirs(full_model_dir)

# Make data directory
#if not os.path.exists(data_dir):
#	os.makedirs(data_dir)



## Download the data if we don't have it saved already.

print('Loading Shakespeare Data')
# Check if file is downloaded.
if not os.path.isfile(os.path.join(data_dir, data_file)):
	print('Not found, downloading Shakespeare texts from www.gutenberg.org')
	'''
	2018-03-30 16:25:19.842690: I tensorflow/core/platform/cpu_feature_guard.cc:137] 
	Loading Shakespeare Data
	Not found, downloading Shakespeare texts from www.gutenberg.org
	[Finished in 1.5s]
	'''
	# print('Not found, downloading Shakespeare texts from \url{www.gutenberg.org}')
	'''
	Loading Shakespeare Data
	Not found, downloading Shakespeare texts from \url{www.gutenberg.org}
	2018-03-30 16:24:33.060713: I tensorflow/core/platform/cpu_feature_guard.cc:137] 
	[Finished in 1.5s]
	'''

	shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
	# Get Shakespeare text
	response = requests.get(shakespeare_url)
	shakespeare_file = response.content
	# Decode binary into string
	s_text = shakespeare_file.decode('utf-8')
	# Drop first few descriptive paragraphs.
	s_text = s_text[7675:]
	# Remove newlines
	s_text = s_text.replace('\r\n', '')
	s_text = s_text.replace('\n', '')

	# Write to file
	with open(os.path.join(data_dir, data_file), 'w') as out_conn:
		out_conn.write(s_text)
	'''
	2018-03-30 17:15:02.280688: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
	Loading Shakespeare Data
	Not found, downloading Shakespeare texts from www.gutenberg.org
	[Finished in 10.4s]
	全部文件只有一行
	'''
else:
	# If file has been saved, load from that file
	with open(os.path.join(data_dir, data_file), 'r') as file_conn:
		s_text = file_conn.read().replace('\n', '')


# Clean text
print('Cleaning Text')
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text) 	# 标点符号字符集
s_text = re.sub('\s+', ' ', s_text) 						# 匹配任意空白字符，一个或多个
print('Done loading/cleaning.')
'''
2018-03-30 17:45:26.516364: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Loading Shakespeare Data
Cleaning Text
Done loading/cleaning.
[Finished in 9.4s]
'''



## Define a function to build a word processing dictionary (word -> ix)

# Build word vocabulary function
def build_vocab(text, min_word_fred):
	word_counts = collections.Counter(text.split(' '))

	# limit word counts to those more frequent than cutoff
	word_counts = {key:val  for key, val in word_counts.items()  if val>min_word_fred}
	# Create vocab --> index mapping
	words = word_counts.keys()
	vocab_to_ix_dict = { key:(ix+1)  for ix, key in enumerate(words) }
	# Add unknown key --> 0 index
	vocab_to_ix_dict['unknown'] = 0
	# Create index --> vocab mapping
	ix_to_vocab_dict = { val:key  for key,val in vocab_to_ix_dict.items() }

	return(ix_to_vocab_dict, vocab_to_ix_dict)


## Now we can build the index-vocabulary from the Shakespeare data.

# Build Shakespeare vocabulary
print('Building Shakespeare Vocab')
ix2vocab, vocab2ix = build_vocab(s_text, min_word_fred) #min_word_freq=5
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert(len(ix2vocab) == len(vocab2ix))

# Convert text to word vectors
s_text_words = s_text.split(' ')
s_text_ix    = []
for ix, x in enumerate(s_text_words):
	try:
		s_text_ix.append( vocab2ix[x] )
	except:
		s_text_ix.append( 0 )
s_text_ix = np.array(s_text_ix)

'''
2018-04-01 17:17:08.252989: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Loading Shakespeare Data
Cleaning Text
Done loading/cleaning.
Building Shakespeare Vocab
Vocabulary Length = 9117
[Finished in 15.2s]
'''




## We define the LSTM model. The methods of interest are the __init__() method, which defines all the model variables and operations, and the sample() method which takes in a sample word and loops through to generate text.

# Define LSTM RNN Model
class LSTM_Model():
	"""docstring for LSTM_Model"""
	def __init__(self, embedding_size, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size, infer_sample=False):
		self.embedding_size = embedding_size
		self.rnn_size 		= rnn_size
		self.vocab_size 	= vocab_size
		self.infer_sample 	= infer_sample
		self.learning_rate 	= learning_rate

		if infer_sample:
			self.batch_size 	  = 1
			self.training_seq_len = 1
		else:
			self.batch_size 	  = batch_size
			self.training_seq_len = training_seq_len

		self.lstm_cell 	   = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
		self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

		self.x_data   = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
		self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

		with tf.variable_scope('lstm_vars'):
			# Softmax Output Weights
			#W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
			#b = tf.get_variable('b', [self.vocab_size], 			   tf.float32, tf.constant_initializer(0.0))
			W = tf.get_variable(name='W', shape=[self.rnn_size, self.vocab_size], initializer=tf.random_normal_initializer())
			b = tf.get_variable(name='b', shape=[self.vocab_size],				  initializer=tf.constant_initializer(0.0))

			# Define Embedding
			#embedding_mat    = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size], tf.float32, tf.random_normal_initializer())
			embedding_mat    = tf.get_variable(name='embedding_mat', shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer())

			embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
			rnn_inputs 		   = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
			rnn_inputs_trimmed = [ tf.squeeze(x, [1])  for x in rnn_inputs]

			'''
			x_data 			tf.int32 	[batch_size, training_seq_len]
			y_output 		tf.int32 	[batch_size, training_seq_len]
			W 				tf.float32 	[rnn_size, vocab_size]
			b 				tf.float32 	[vocab_size,]
			embedding_mat 	tf.float32 	[vocab_size, embedding_size]

			embedding_output 	tf.float32 	[batch_size, training_seq_len, embedding_size]
			rnn_inputs 			tf.float32 	``training_seq_len'' items, each ``[batch_size, 1, embedding_size]''
			rnn_inputs_trimmed 	tf.float32 	``training_seq_len'' items, each ``[batch_size, embedding_size''

			Method 2:
			no ``rnn_inputs''
			rnn_inputs_trimmed = tf.unstack(value=embedding_output, num=self.training_seq_len, axis=1)
			Note: 	tf.squeeze(rnn_inputs[i], axis=[1])
			'''

		# If we are inferring (generating text), we add a `loop' function
		# Define how to get the i+1 th input from the i th output
		def inferred_loop(prev, count):
			# Apply hidden layer
			prev_transformed = tf.matmul(prev, W) + b 
			# Get the index of the output (also don't run the gradient)
			prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
			# Get embedded vector
			output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
			return(output)
			'''
			prev_symbol 		
					tf.argmax(prev_transformed, axis=1)

			W 				tf.float32 	[rnn_size, vocab_size]
			b 				tf.float32 	[vocab_size,]
			embedding_mat 	tf.float32 	[vocab_size, embedding_size]
			'''

		decoder = tf.contrib.legacy_seq2seq.rnn_decoder
		outputs, last_state = decoder(rnn_inputs_trimmed, 
									  self.initial_state, 
									  self.lstm_cell, 
									  loop_function=inferred_loop if infer_sample else None)
		# Non inferred outputs
		output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
		'''
		rnn_inputs_trimmed 		tf.float32 	``training_seq_len'' items, each ``[batch_size, embedding_size''
		initial_state 			tf.float32 	
			outputs 			tf.float32 	``training_seq_len'' items, each ``[batch_size, ]''
			last_state 			tf.floate 	
		'''
		# Logits and output
		self.logit_output = tf.matmul(output, W) + b 
		self.model_output = tf.nn.softmax(self.logit_output)

		loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
		loss     = loss_fun([self.logit_output], 
							[tf.reshape(self.y_output, [-1])], 
							[tf.ones([self.batch_size * self.training_seq_len])], 
							self.vocab_size)
		self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
		self.final_state = last_state
		gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
		optimizer    = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

	def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
		state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
		word_list = prime_text.split()
		for word in word_list[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = vocab[word]
			feed_dict = {self.x_data: x, self.initial_state: state}
			[model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
			sample = np.argmax(model_output[0])
			if sample == 0:
				break
			word = words[sample]
			out_sentence = out_sentence + ' ' + word
		return(out_sentence)


## In order to use the same model (with the same trained variables), we need to share the variable scope between the trained model and the test model.

# Define LSTM Model
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
	test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size, infer_sample=True)


## We need to save the model, so we create a model saving operation.

# Create model saver
saver = tf.train.Saver(tf.global_variables())









