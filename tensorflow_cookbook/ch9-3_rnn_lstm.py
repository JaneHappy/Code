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



# Download the data if we don't have it saved already.

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



# Define a function to build a word processing dictionary (word -> ix)

# Build word vocabulary function
def build_vocab(text, min_word_fred):
	word_counts = collections.Counter(text.split(' '))

	# limit word counts to those more frequent than cutoff
	word_counts = {key:val  for key, val in word_counts.items()  if val>min_word_fred}





