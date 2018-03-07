# coding: utf-8
# Implementing an RNN in TensorFlow

from __future__ import division
from __future__ import print_function


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
sess = tf.Session()


# Set RNN parameters
epochs 				= 20
batch_size 			= 250
max_sequence_length = 25
rnn_size 			= 10
embedding_size 		= 50
min_word_frequency 	= 10
learning_rate 		= 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)



# Download or open data
data_dir  = ''  #'temp'
data_file = 'text_data.txt'
#if not os.path.exists(data_dir):
#	os.makedirs(data_dir)

if not os.path.isfile(os.path.join(data_dir, data_file)):
	zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
	r = requests.get(zip_url)
	z = ZipFile(io.BytesIO(r.content))
	file = z.read('SMSSpamCollection')
	print('r,z,file', r,z, '\n')  #print('r,z,file,', r, z, file, '\n')
	# Format Data
	text_data = file.decode('utf8')  ##file.decode()
	#print(text_data, '\n')
	text_data = text_data.encode('ascii', errors='ignore')
	#print(text_data)
	text_data = text_data.decode().split('\n')
	print(text_data, '\n')

	# Save data to text file
	with open(os.path.join(data_dir, data_file), 'w') as file_conn:
		for text in text_data:
			file_conn.write("{}\n".format(text))
else:
	# open data from text file
	text_data = []
	with open(os.path.join(data_dir, data_file), 'r') as file_conn:
		for row in file_conn:
			text_data.append(row)
	text_data = text_data[:-1]  #the last one is '\n'


text_data = [ x.split('\t')  for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [ list(x)  for x in zip(*text_data)]


'''
2018-03-06 15:21:55.571143: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
r,z,file <Response [200]> <zipfile.ZipFile object at 0x7f69c14c12d0> 

[u'ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...', ..., u'ham\tRofl. Its true to its name', u''] 

[Finished in 4.3s]


>>> text_data 
 ...
 suggestions?\n', "ham\tThe guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n", 'ham\tRofl. Its true to its name\n', '\n']
>>> text_data = text_data[:-1]
>>> text_data[0]
'ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n'
>>> text_data[0].split('\t')
['ham', 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n']
>>> 

'''



# Create a text cleaning function
def clean_text(text_string):
	text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
	text_string = " ".join(text_string.split())
	text_string = text_string.lower()
	return(text_string)

# Clean texts
text_data_train = [ clean_text(x)  for x in text_data_train]

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_word_frequency)
text_processed  = np.array(list( vocab_processor.fit_transform(text_data_train) ))

# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x=='ham' else 0  for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[: ix_cutoff], x_shuffled[ix_cutoff :]
y_train, y_test = y_shuffled[: ix_cutoff], y_shuffled[ix_cutoff :]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# Create placeholders
x_data   = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# Create embedding
embedding_mat    = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
embedding_output_expanded = tf.expand_dims(embedding_output, -1)

'''
print('embedding_mat\t\t', embedding_mat)
print('embedding_output\t\t', embedding_output)
print('embedding_output_expanded\t', embedding_output_expanded)

2018-03-06 16:24:12.005208: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Vocabulary Size: 933
80-20 Train Test split: 4459 -- 1115
embedding_mat				 <tf.Variable 'Variable:0' 	  shape=(933, 50) 		dtype=float32_ref>
embedding_output			 Tensor("embedding_lookup:0", shape=(?, 25, 50), 	dtype=float32)
embedding_output_expanded	 Tensor("ExpandDims:0", 	  shape=(?, 25, 50, 1), dtype=float32)
[Finished in 26.6s]





>>> train[0]
'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n'
>>> import re
>>> re.sub(r'([^\s\w]|_|[0-9])+', '', train[0])
'Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amore wat\n'
>>> re.sub(r'([^\s\w]|_|[0-9])+', '', train[])
  File "<stdin>", line 1
    re.sub(r'([^\s\w]|_|[0-9])+', '', train[])
                                            ^
SyntaxError: invalid syntax
>>> train[1]
'Ok lar... Joking wif u oni...\n'
>>> re.sub(r'([^\s\w]|_|[0-9])+', '', train[1])
'Ok lar Joking wif u oni\n'
>>> 

>>> a = re.sub(r'([^\s\w]|_|[0-9])+', '', train[1])
>>> " ".join(a.split())
'Ok lar Joking wif u oni'
>>> a
'Ok lar Joking wif u oni\n'
>>> a.split()
['Ok', 'lar', 'Joking', 'wif', 'u', 'oni']
>>> a.strip()
'Ok lar Joking wif u oni'
>>> 

>>> a1= " ".join(a.split())
>>> a1
'Ok lar Joking wif u oni'
>>> a1.lower()
'ok lar joking wif u oni'
>>> 

>>> from tensorflow.contrib import learn
>>> x_text = ['i love you', 'me too']
>>> vocab_pro = learn.preprocessing.VocabularyProcessor(10)
>>> vocab_pro.fit(x_text)
<tensorflow.contrib.learn.python.learn.preprocessing.text.VocabularyProcessor object at 0x7f00a90909d0>
>>> print next(vocab_pro.transform(['i me too'])).tolist()
[1, 4, 5, 0, 0, 0, 0, 0, 0, 0]
>>> x = np.array(list(vocab_pro.fit_transform(x_text)))
>>> x
array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
       [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]])
>>> vocab_pro.vocabulary_
<tensorflow.contrib.learn.python.learn.preprocessing.categorical_vocabulary.CategoricalVocabulary object at 0x7f00a912da10>
>>> len( vocab_pro.vocabulary_ )
6
>>> 

>>> vocab_pro.fit_transform(x_text)
<generator object transform at 0x7f008d3a5f00>
>>> list( vocab_pro.fit_transform(x_text) )
[array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0]), array([4, 5, 0, 0, 0, 0, 0, 0, 0, 0])]
>>> np.array( list( vocab_pro.fit_transform(x_text) ))
array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
       [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]])
>>>

>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.random.permutation( np.arange(10))
array([2, 5, 6, 7, 0, 8, 1, 3, 4, 9])
>>> 

>>> s.run(tf.random_uniform([3,2], -1.0, 1.0))
array([[ 0.01749253,  0.34167552],
       [ 0.07842255, -0.49839091],
       [-0.92785692,  0.21439958]], dtype=float32)


'''



# Define the RNN cell
# tensorflow change >= 1.0, rnn is put into tensorflow.contrib.directory. Prior version not test.
if tf.__version__[0] >= '1':
	cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
else:
	cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)








'''
References:
https://github.com/nfmcclure/tensorflow_cookbook/blob/master/09_Recurrent_Neural_Networks/02_Implementing_RNN_for_Spam_Prediction/02_implementing_rnn.py
'''
