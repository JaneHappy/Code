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
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
#embedding_output_expanded = tf.expand_dims(embedding_output, -1)

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
output = tf.nn.dropout(output, dropout_keep_prob)







