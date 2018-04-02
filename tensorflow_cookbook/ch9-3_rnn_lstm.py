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
epochs 			 = 2 #10 		# Number of epochs to cycle through data
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
		s_text_ix.append( 0 ) 	#s_text_ix.append( vocab2ix['unknown'] )
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
			[state] = sess.run([self.final_state], feed_dict=feed_dict)

		out_sentence = prime_text
		word = word_list[-1]
		for n in range(num):
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
'''
embedding_size 	 = 100 
rnn_size 		 = 128 
embedding_size 	 = rnn_size
batch_size 		 = 100 
learning_rate 	 = 0.001 
training_seq_len = 50 
vocab_size 		 = len(ix2vocab) + 1 	= 9117
'''

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
	test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size, infer_sample=True)


## We need to save the model, so we create a model saving operation.

# Create model saver
saver = tf.train.Saver(tf.global_variables())

## Let's calculate how many batches are needed for each epoch and split up the data accordingly.

# Create batches for each epoch
num_batches = int(len(s_text_ix) / (batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

'''
>>> batch_size = 100
>>> training_seq_len = 50
>>> num_batches = int(len(s_text_ix) / (float(batch_size) * training_seq_len)) + 1
>>> num_batches
181
>>> s_text_ix.shape
(901089,)
>>> batches = np.array_splits(s_text_ix, num_batches)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'array_splits'
>>> batches = np.array_split(s_text_ix, num_batches)
>>> batches2 = [np.resize(x, [batch_size, training_seq_len])  for x in batches]

>>> batches[0]
array([6497, 4191, 7136, ..., 3019, 4868, 6969])
>>> batches2[0]
array([[6497, 4191, 7136, ..., 6965, 3027, 3667],
       [1379, 2675,  484, ..., 4492, 2684, 2338],
       [8376, 1621, 2684, ..., 5200, 8014, 2684],
       ..., 
       [1455, 1451, 8014, ..., 1074, 2675,  369],
       [   0, 8706, 5254, ..., 6949, 1455, 5299],
       [ 971, 1455, 2405, ..., 1621,  581,    0]])
>>> batches[0].shape
(4979,)
>>> batches2[0].shape
(100, 50)
>>> 

>>> for it,t in enumerate(batches):
...     if it%10 ==9:
...             print t.shape
...     else:
...             print t.shape ,
...
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)
(4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,) (4979,)

(4979,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,) (4978,)
(4978,)

'''


## Initialize all the variables

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)


## Train the model!

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
	# Shuffle word indices
	random.shuffle(batches)
	# Create targets from shuffled batches
	targets = [np.roll(x, -1, axis=1)  for x in batches]
	'''
	#一个单词是一个样本
	
	s_text_ix 	是 s_text 中的这么多个单词、分别对应的索引，即 vocab2ix[单词]
	batches = np.array_split(s_text_ix, ..) 
		把 s_text_ix 划分成了 num_batches=181 个小批次，
		每一批次有接近 batch_size * training_seq_len =5000 个单词，
		这 5000 个单词可以看成是 batch_size=100 个样本即句子，
		每个样本/句子中包含 training_seq_len=50 个单词 —— 实际这里是 vocab2ix[单词]
	batches = [np.resize(x, [batch_size, training_seq_len])  for x in batches]
		一共有 num_batches=181 个批次。
		在每一个批次中，把这一批次的接近 5000 个样本，重新组织，
		变成 100 个样本/句子，每个句子包含 50 个单词 —— 实际这里是 vocab2ix[单词]
	
	epochs  = 10
	batches = a list, with ``num_batches=181'' items, each ``[batch_size, training_seq_len''
	iteration_count = 1~~(epoches*num_batches = 10*181=1810)
	'''

	# Run a through one epoch
	print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
	# Reset initial LSTM state every epoch
	state = sess.run(lstm_model.initial_state)
	for ix, batch in enumerate(batches):
		training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
		c, h = lstm_model.initial_state
		training_dict[c] = state.c 
		training_dict[h] = state.h 

		temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op], feed_dict=training_dict)
		train_loss.append(temp_loss)

		# Print status every 10 gens
		if iteration_count % 10 == 0:
			#summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
			#print('Iterations: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
			summary_nums = (iteration_count, epoch+1, ix+1, num_batches, temp_loss)
			print('\tIterations: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.6f}'.format(*summary_nums))

		# Save the model and the vocab
		if iteration_count % save_every == 0:
			# Save model
			model_file_name = os.path.join(full_model_dir, 'model')
			saver.save(sess, model_file_name, globel_step = iteration_count)
			#print('Model Saved To: {}'.format(model_file_name))
			print('\t\tModel Saved To: {} .'.format(model_file_name))
			# Save vocabulary
			dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
			with open(dictionary_file, 'wb') as dict_file_conn:
				pickle.dump([vocab2ix, ix2vocab], dict_file_conn)

		if iteration_count % eval_every == 0:
			for sample in prime_texts:
				print('\t\t\tPredict:\t', end="")
				print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))

		iteration_count += 1




'''
2018-04-02 12:06:42.238047: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Loading Shakespeare Data
Cleaning Text
Done loading/cleaning.
Building Shakespeare Vocab
Vocabulary Length = 9117
Starting Epoch #1 of 2.
	Iterations: 10, Epoch: 1, Batch: 10 out of 181, Loss: 10.007343
	Iterations: 20, Epoch: 1, Batch: 20 out of 181, Loss: 9.179014
	Iterations: 30, Epoch: 1, Batch: 30 out of 181, Loss: 8.776321
	Iterations: 40, Epoch: 1, Batch: 40 out of 181, Loss: 8.622705
	Iterations: 50, Epoch: 1, Batch: 50 out of 181, Loss: 8.414643
			Predict:	thou art more hand- like leaden entertain'd made when the
			Predict:	to be or not to the
			Predict:	wherefore art thou Sex Muse and
	Iterations: 60, Epoch: 1, Batch: 60 out of 181, Loss: 8.034399
	Iterations: 70, Epoch: 1, Batch: 70 out of 181, Loss: 7.787870
	Iterations: 80, Epoch: 1, Batch: 80 out of 181, Loss: 7.644072
	Iterations: 90, Epoch: 1, Batch: 90 out of 181, Loss: 7.622431
	Iterations: 100, Epoch: 1, Batch: 100 out of 181, Loss: 7.494274
			Predict:	thou art more than often
			Predict:	to be or not to the
			Predict:	wherefore art thou At going my lord I am
	Iterations: 110, Epoch: 1, Batch: 110 out of 181, Loss: 7.717870
	Iterations: 120, Epoch: 1, Batch: 120 out of 181, Loss: 7.654597
	Iterations: 130, Epoch: 1, Batch: 130 out of 181, Loss: 7.711281
	Iterations: 140, Epoch: 1, Batch: 140 out of 181, Loss: 7.740433
	Iterations: 150, Epoch: 1, Batch: 150 out of 181, Loss: 7.305348
			Predict:	thou art more than the
			Predict:	to be or not to the
			Predict:	wherefore art thou art too cause to the
	Iterations: 160, Epoch: 1, Batch: 160 out of 181, Loss: 6.868250
	Iterations: 170, Epoch: 1, Batch: 170 out of 181, Loss: 6.693734
	Iterations: 180, Epoch: 1, Batch: 180 out of 181, Loss: 6.817922
Starting Epoch #2 of 2.
	Iterations: 190, Epoch: 2, Batch: 9 out of 181, Loss: 6.632661
	Iterations: 200, Epoch: 2, Batch: 19 out of 181, Loss: 6.494490
			Predict:	thou art more than the
			Predict:	to be or not to the
			Predict:	wherefore art thou art part no no more than the
	Iterations: 210, Epoch: 2, Batch: 29 out of 181, Loss: 6.605542
	Iterations: 220, Epoch: 2, Batch: 39 out of 181, Loss: 6.852113
	Iterations: 230, Epoch: 2, Batch: 49 out of 181, Loss: 6.654508
[Cancelled]
'''





