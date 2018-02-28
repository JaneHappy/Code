# coding: utf-8
# tensorflow_cookbook
#	01 Introduction
#		07 Working with Data Sources
#	https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/07_Working_with_Data_Sources/07_data_gathering.ipynb




from __future__ import print_function

# Data Gathering
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops
ops.reset_default_graph()


## The Iris Dataset
from sklearn.datasets import load_iris
iris = load_iris()
print("len(data  )", len(iris.data  ))
print("len(target)", len(iris.target))
print("  data[0]  ", iris.data[0] )
print("  target   ", set(iris.target))
'''
len(data  ) 150
len(target) 150
  data[0]   [ 5.1  3.5  1.4  0.2]
  target    set([0, 1, 2])
[Finished in 1.7s]
'''


## Low Birthrate Dataset (Hosted on Github)
import requests 
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file    = requests.get(birthdata_url) 	#[Finished in 3.2s], without file download
print(birth_file)
birth_data    = birth_file.text.split('\r\n')
birth_header  = birth_data[0].split('\t')
birth_d_prime = [ [ float(x)  for x in y.split('\t')]  for y in birth_data[1:] if len(y)>=1]
print("birth_header", birth_header)
print("birth_data (raw): length ", len(birth_data[1:])   )
print("birth_data      : length ", len(birth_d_prime)    )
print("birth_data[0]   : length ", len(birth_d_prime[0]) )

print("birth_header    : ", birth_header )
print("birth_data[0]   : ", birth_d_prime[0] )
'''
<Response [200]>
birth_header [u'LOW', u'AGE', u'LWT', u'RACE', u'SMOKE', u'PTL', u'HT', u'UI', u'BWT']
birth_data (raw): length 191
birth_data      : length 189
birth_data[0]   : length 9
[Finished in 3.1s]

birth_header    :  [u'LOW', u'AGE', u'LWT', u'RACE', u'SMOKE', u'PTL', u'HT', u'UI', u'BWT']
birth_data[0]   :  [1.0, 28.0, 113.0, 1.0, 1.0, 1.0, 0.0, 1.0, 709.0]
[Finished in 2.5s]
'''


## Housing Price Dataset (UCI)
import requests
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']	#Attribute Information:
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names
housing_file = requests.get(housing_url)
housing_data = housing_file.text.split('\n')
#or: housing_prime = [ [float(x)  for x in y.split(' ') if len(x)>=1]  for y in housing_data if len(y)>=1]
housing_prime = [[float(x)  for x in y.split()]  for y in housing_data if len(y)>0]
print("len(housing_data )    ", len(housing_data ) )
print("len(housing_prime)    ", len(housing_prime) )
print("len(housing_prime[0]) ", len(housing_prime[0]) )
'''
len(housing_data )     507
len(housing_prime)     506
len(housing_prime[0])  14
[Finished in 5.3s]
'''


# MNIST Handwriting Dataset (Yann LeCun)
from tensorflow.examples.tutorials.mnist import input_data
#or: mnist = input_data.read_data_sets("MNIST_data/", onehot=True)
mnist = input_data.read_data_sets("", one_hot=True)
print("mnist.train.images		", len(mnist.train.images))
print("mnist.test.images 		", len(mnist.test.images ))
print("mnist.validation.images	", len(mnist.validation.images))
print("mnist.train.labels[1,:]	", mnist.train.labels[1,:])
'''
Extracting train-images-idx3-ubyte.gz
Extracting train-labels-idx1-ubyte.gz
Extracting t10k-images-idx3-ubyte.gz
Extracting t10k-labels-idx1-ubyte.gz
mnist.train.images		 55000
mnist.test.images 		 10000
mnist.validation.images	 5000
mnist.train.labels[1,:]	 [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
[Finished in 27.6s]
'''
print("mnist.train.images 		", mnist.train.images.shape)
print("mnist.train.labels 		", mnist.train.labels.shape)
print("mnist.test.images 		", mnist.test.images.shape )
print("mnist.test.labels 		", mnist.test.labels.shape )
print("mnist.validation.images 	", mnist.validation.images.shape)
print("mnist.validation.labels 	", mnist.validation.labels.shape)
'''
mnist.train.images 		 (55000, 784)
mnist.train.labels 		 (55000,  10)
mnist.test.images 		 (10000, 784)
mnist.test.labels 		 (10000,  10)
mnist.validation.images  ( 5000, 784)
mnist.validation.labels  ( 5000,  10)
[Finished in 13.0s]
'''


# CIFAR-10 Data
from PIL import Image 
# Running this command requires an internet connection and a few minutes to download all the images.
(X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
# Downloading data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# The ten categories are (in order):
# 1.Airplane 2.Automobile 3.Bird 4.Car 5.Deer 6.Dog 7.Frog 8.Horse 9.Ship 10.Truck
print("x_trn ", X_train.shape)
print("y_trn ", y_train.shape)
print("x_tst ", X_test.shape )
print("y_tst ", y_test.shape )
print("y_trn[0] ", y_train[0]) #or: y_train[0,])
'''
x_trn  (50000, 32, 32, 3)
y_trn  (50000, 1)
x_tst  (10000, 32, 32, 3)
y_tst  (10000, 1)
y_trn[0]  [6]
[Finished in 49.8s]
'''

# Plot the 0-th image (a frog)
#%matplotlib inline
#or: img = Image.fromarray(X_train[0, :,:,:])
img = Image.fromarray(X_train[0])
plt.imshow(img)
'''
plt.show()
[Finished in 24.9s]
'''


# Ham/Spam Texts Dataset (UCI)
import requests
import io 
from zipfile import ZipFile 
# Get/read zip file
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('SMSSpamCollection')
# Format Data
''' error!
text_data = file.decode()
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2 in position 571: ordinal not in range(128)
https://stackoverflow.com/questions/16508539/unicodedecodeerror-ascii-codec-cant-decode-byte-0xc2
'''
text_data = file.decode('utf8')
text_data = text_data.encode('ascii', errors='ignore')
text_data = text_data.decode().split('\n')
text_prime = [ x.split('\t')  for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [ list(x)  for x in zip(*text_prime)]
print("len(text_data) raw :  ", len(text_data ))
print("len(text_prime)    :  ", len(text_prime))
print("len(text_data_train ) ", len(text_data_train ))
print("len(text_data_target) ", len(text_data_target))
print("set(text_data_target) ", set(text_data_target))
print("  text_data_target[1] ", text_data_target[1])
print("  text_data_train[ 1] ", text_data_train[ 1])
print("  text_data_target[:1]", text_data_target[:2])
print("  text_data_train[ :1]", text_data_train[ :2])
'''
len(text_data) raw :   5575
len(text_prime)    :   5574
len(text_data_train )  5574
len(text_data_target)  5574
set(text_data_target)  set([u'ham', u'spam'])
  text_data_target[1]  ham
  text_data_train[ 1]  Ok lar... Joking wif u oni...
[Finished in 34.5s]
  text_data_target[:1] [u'ham']
  text_data_train[ :1] [u'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...']
[Finished in 27.5s]
  text_data_target[:2] [u'ham', u'ham']
  text_data_train[ :2] [u'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...', u'Ok lar... Joking wif u oni...']
[Finished in 24.8s]
'''


# Movie Review Data (Cornell)
import requests
import io
import tarfile
movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)
# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
	s = stream_data.read(16384)
	if not s:
		break 
	tmp.write(s)
stream_data.close()
tmp.seek(0)
# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
# Save pos/neg reviews
pos_data = []
for line in pos:
	pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
neg_data = []
for line in neg:
	neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
tar_file.close()
print("len(pos_data) ", len(pos_data))
print("len(neg_data) ", len(neg_data))
print("  pos_data[0] ", pos_data[0])
print("  neg_data[0] ", neg_data[0])
'''
len(pos_data)  5331
len(neg_data)  5331
  pos_data[0]  the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . 

  neg_data[0]  simplistic , silly and tedious . 

[Finished in 16.0s]

>>> pos_data[0]
u'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . \n'
>>> neg_data[0]
u'simplistic , silly and tedious . \n'
>>> u(neg_data[0])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'u' is not defined
>>> 
'''


# The Complete Works of William Shakespeare (Gutenberg Project)
# The Works of Shakespeare Data
import requests
shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content
# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')
# Drop first few descriptive paragraphs.
shakespeare_prime = shakespeare_text[7675:]
print(len(shakespeare_prime))
print("len(shakespeare_text[7675:]) ", len(shakespeare_text[7675:]))
print("len(shakespeare_text[:7675]) ", len(shakespeare_text[:7675]))
print("  shakespeare_text[0]    ", shakespeare_text[   0])
print("  shakespeare_text[7675] ", shakespeare_text[7675])
'''
5582212
len(shakespeare_text[7675:])  5582212
len(shakespeare_text[:7675])  7675
  shakespeare_text[0]     ﻿
  shakespeare_text[7675]  F
[Finished in 35.8s]
'''


# English-German Sentence Translation Database (Manythings/Tatoeba)
# English-German Sentence Translation Data
import requests
import io
from zipfile import ZipFile
sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentence_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('deu.txt')
# Format Data
#error! eng_ger_data = file.decode()
eng_ger_data = file.decode('utf8')
eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_prime = [ x.split('\t')  for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_prime)]
print("len(eng_ger_data) raw : ", len(eng_ger_data ))
print("len(eng_ger_data)     : ", len(eng_ger_prime))
print("len(english_sentence) : ", len(english_sentence))
print("len( german_sentence) : ", len( german_sentence))
print("	eng_ger_data[10]	 ", eng_ger_data[10])
print(" english_sentence[:10]", english_sentence[:10])
print("  german_sentence[:10]",  german_sentence[:10])
'''
len(eng_ger_data) raw :  159205
len(eng_ger_data)     :  159204
len(english_sentence) :  159204
len( german_sentence) :  159204
	eng_ger_data[10]	  Hello!	Hallo!
 english_sentence[:10] [u'Hi.', u'Hi.', u'Run!', u'Wow!', u'Wow!', u'Fire!', u'Help!', u'Help!', u'Stop!', u'Wait!']
  german_sentence[:10] [u'Hallo!', u'Gr Gott!', u'Lauf!', u'Potzdonner!', u'Donnerwetter!', u'Feuer!', u'Hilfe!', u'Zu Hlf!', u'Stopp!', u'Warte!']
[Finished in 43.3s]
'''

