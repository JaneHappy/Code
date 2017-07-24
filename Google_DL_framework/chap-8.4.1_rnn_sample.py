# coding: utf-8

#  chapter 8.4    examples of RNN, PTB dataset
#    Python packages:    /usr/local/lib/python2.7/dist-packages

#  References:
#    https://github.com/caicloud/tensorflow-tutorial/blob/master/caicloud.tensorflow/caicloud/clever/examples/ptb/grpc_client.py
#    https://github.com/caicloud/tensorflow-tutorial/tree/master/caicloud.tensorflow/caicloud/clever/examples/ptb




'''
from tensorflow.models.rnn.ptb import reader
ImportError: No module named models.rnn.ptb
'''
import tensorflow as tf 
import ptb_word_lm
import reader


# 存放原始数据的路径。
##DATA_PATH = "/path/to/ptb/data"
DATA_PATH = "/home/ubuntu/Program/GoogleDL_datasets/simple-examples/data/"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
# 读取原始数据。
##print len(train_data)
print "PTB dataset: \t train %d,\t valid %d, \t test %d." % (len(train_data), len(valid_data), len(test_data))
print train_data[:100]

'''
运行以上程序可以得到输出：

PTB dataset: 	 train 929589,	 valid 73760, 	 test 82430.
[9970, 9971, 9972, 9974, 9975, 9976, 9980, 9981, 9982, 9983, 9984, 9986, 9987, 9988, 9989, 9991, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999, 2, 9256, 1, 3, 72, 393, 33, 2133, 0, 146, 19, 6, 9207, 276, 407, 3, 2, 23, 1, 13, 141, 4, 1, 5465, 0, 3081, 1596, 96, 2, 7682, 1, 3, 72, 393, 8, 337, 141, 4, 2477, 657, 2170, 955, 24, 521, 6, 9207, 276, 4, 39, 303, 438, 3684, 2, 6, 942, 4, 3150, 496, 263, 5, 138, 6092, 4241, 6036, 30, 988, 6, 241, 760, 4, 1015, 2786, 211, 6, 96, 4]
[Finished in 5.2s]
'''





# 为了实现截断并将数据组织成 batch，TensorFlow 提供了 ptb_iterator 函数。

# 类似地读取数据原始数据。
## DATA_PATH = 
##  = reader.ptb_raw_data(DATA_PATH)

# 将训练数据组织成 batch 大小为4、截断长度为5的数据组。
##result = reader.ptb_iterator(train_data, 4, 5)
# 读取第一个batch中的数据，其中包括每个时刻的输入和对应的正确输出。
##x, y = result.next()

x,y = reader.ptb_producer(train_data, 4, 5)
print "X: ", x
print "y: ", y

'''
运行以上程序可以得到输出：

X:  Tensor("PTBProducer/StridedSlice:0", shape=(4, 5), dtype=int32)
y:  Tensor("PTBProducer/StridedSlice_1:0", shape=(4, 5), dtype=int32)
[Finished in 5.6s]
'''






#---------------------------------------
# |->  2017.7.23
#
# |->  @ 2017.7.24
#  References:
#  http://tensorlayer.readthedocs.io/en/latest/modules/iterate.html
#---------------------------------------
'''
import tensorlayer as tl 

batch = tl.iterate.ptb_iterator(train_data, batch_size=4, num_steps=5)
x, y = batch
print "X ---> ", x
print "y ---> ", y

Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.1_rnn_sample.py", line 82, in <module>
    x, y = batch
ValueError: too many values to unpack
[Finished in 6.1s]
'''




