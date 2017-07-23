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


'''
















