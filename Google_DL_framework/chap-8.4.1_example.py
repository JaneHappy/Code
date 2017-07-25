# -*- coding: utf-8 -*-
#  chapter 8.4.1  
#  References:
#  https://www.tensorflow.org/tutorials/word2vec




import numpy as np 

import tensorflow as tf 
# from tensorflow.models.rnn.ptb import reader
import reader


# DATA_PATH = "/path/to/ptb/data" 	# 数据存放的路径
DATA_PATH = "/home/ubuntu/Program/GoogleDL_datasets/simple-examples/data"
HIDDEN_SIZE = 200 					# 隐藏层规模
NUM_LAYERS = 2 						# 深层循环神经网络中LSTM结构的层数。
VOCAB_SIZE = 10000 					# 词典规模，加上语句结束标识符和稀有单词标识符总共一万个单词。

LEARNING_RATE = 1.0 				# 学习速率。
TRAIN_BATCH_SIZE = 20 				# 训练数据batch的大小。
TRAIN_NUM_STEP = 35 				# 训练数据截断长度。


# 在测试时不需要使用截断，所以可以将测试数据看成一个超长的序列。
EVAL_BATCH_SIZE = 1 		# 测试数据batch的大小。
EVAL_NUM_STEP = 1 			# 测试数据截断长度。
NUM_EPOCH = 2 				# 使用训练数据的轮数。
KEEP_PROB = 0.5 			# 节点不被dropout的概率。
MAX_GRAD_NORM = 5 			# 用于控制梯度膨胀的参数。


# 通过一个 PTBModel 类来描述模型，这样方便维护循环神经网络中的状态。
class PTBModel(object):
	"""docstring for PTBModel"""
	def __init__(self, is_training, batch_size, num_steps):
		# 记录使用的batch大小和截断长度。
		self.batch_size = batch_size
		self.num_steps = num_steps

		# 定义输入层。可以看到输入层的维度为 batch_size x num_steps，这和 ptb_iterator 函数输出的训练数据 batch 是一致的。
		self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

		# 定义预期输出。它的维度和 ptb_iterator 函数输出的正确答案维度也是一样的。
		self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

		# 定义使用LSTM结构为循环体结构且使用 dropout 的深层循环神经网络。
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
		if is_training :
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_)
# 初始化最初的状态，也就是全零的向量。
# 将单词ID转换成为单词向量。因为总共有 VOCAB_SIZE 个单词，每个单词向量的维度为 HIDDEN_SIZE，所以 embedding 参数的维度为 VOCAB_SIZE x HIDDEN_SIZE 。
# 将原本 batch_size x num_steps 个单词ID转化为单词向量，转化后的输入层维度为 batch_size x num_steps x HIDDEN_SIZE。
# 只在训练时使用 dropout。
# 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再通过一个全连接层得到最终的输出。
# state 存储不同batch中LSTM的状态，将其初始化为0.
# 从输入数据中获取当前时刻获得输入并传入LSTM结构。
# 将当前输出加入输出队列。

# 把输出队列展开成 [batch, hidden_size*num_steps] 的形状，然后再 reshape 成 [batch*numsteps, hidden_size] 的形状。

# 将从LSTM中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在每一个时刻上都是一个长度为 VOCAB_SIZE 的数组，经过 softmax 层之后表示下一个位置是不同单词的概率。

# 定义交叉熵损失函数。TensorFlow提供了 sequence_loss_by_example 函数来计算一个序列的交叉熵的和。
# 预测的结果。
# 期待的正确答案，这里将 [batch_size, num_steps] 二维数组压缩成一维数组。
# 损失的权重。在这里所有的权重都为1，也就是说不同batch和不同时刻的重要程度是一样的。
# 计算得到每个batch的平均损失。
# 只在训练模型时定义反向传播操作。
# 通过 clip_by_global_norm 函数控制梯度的大小，避免梯度膨胀的问题。
# 定义优化方法。
# 定义训练步骤。






