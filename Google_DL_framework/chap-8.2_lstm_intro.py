# coding: utf-8
#  chapter 8.2  LSTM intro
#  chapter 8.3  variation: bidirectional RNN, deepRNN




#---------------------------------------
# 以下代码展示了在TensorFlow中实现使用LSTM结构的循环神经网络的前向传播过程。
#---------------------------------------


# 定义一个 LSTM 结构。在 TensorFlow 中通过一句简单的命令就可以实现一个完整 LSTM 结构。LSTM 中使用的变量也会在该函数中自动被声明。
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 将 LSTM 中的状态初始化为全0数组。和其他NN类似，在优化循环神经网络时，每次也会使用一个 batch 的训练样本。以下代码中，batch_size 给出了一个 batch 的大小。BasicLSTMCell 类提供了 zero_state 函数来生成全零的初始状态。
state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数。
loss = 0.0
# 在8.1节中介绍过，虽然理论上RNN可以处理任意长度的序列，但是在训练时为了避免梯度消散的问题，会规定一个最大的序列长度。在以下代码中，用 num_steps 来表示这个长度。
for i in range(num_steps):
	# 在第一个时刻声明 LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量。
	if i > 0:	tf.get_variable_scope().reuse_variables()
	# 每一步处理时间序列中的一个时刻。将当前输入 (current_input) 和前一时刻状态 (state) 传入定义好的LSTM结构可以得到当前LSTM结构的输出 lstm_output 和更新后的状态 state。
	lstm_output, state = lstm(current_input, state)
	# 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出。
	final_output = fully_connected(lstm_output)
	# 计算当前时刻输出的损失。
	loss += calc_loss(final_output, expected_output)

# 使用类似第4章中介绍的方法训练模型。







#----------------------------------------------
'''
# References:
https://www.tensorflow.org/tutorials/recurrent
https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537
https://liusida.github.io/2016/11/16/study-lstm/
http://www.infoq.com/cn/news/2016/07/TensorFlow-LSTM
'''



'''
import tensorflow as tf 

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
state = tf.zeros([batch_size, lstm.state_size])
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
	# The value of state is updated after processing each batch of words.
	output, state = lstm(current_batch_of_words, state)
	# The LSTM output can be used to make next word predictions
	logits = tf.matmul(output, softmax_w) + softmax_b
	probabilities.append(tf.nn.softmax(logits))
	loss += loss_function(probabilities, target_words)

'''





#-----------------------------------
#  chapter 8.3 
#-----------------------------------




# 定义一个基本的LSTM结构作为循环体的基础结构。深层循环神经网络也支持使用其他的循环体结构。
lstm = rnn_cell.BasicLSTMCell(lstm_size)
# 通过 MultiRNNCell 类实现 deepRNN 中每一个时刻的前向传播过程。其中 number_of_layers 表示了有多少层，也就是 图8-16 中从 xt 到 ht 需要经过多少个 LSTM 结构。
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

# 和经典的RNN一样，可以通过 zero_state 函数来获取初始状态。
state = stacked_lstm.zero_state(batch_size, tf.float32)

# 和8.2节中给出的代码一样，计算每一时刻的前向传播结果。
for i in range(len(num_steps)):
	if i> 0:	tf.get_variable_scope().reuse_variables()
	stacked_lstm_output, state = stacked_lstm(current_input, state)
	final_output = fully_connected(stacked_lstm_output)
	loss += calc_loss(final_output, expected_output)




# 定义LSTM结构。
lstm = rnn_cell.BasicLSTMCell(lstm_size)

# 使用 DropoutWrapper 类来实现 dropout 功能。该类通过两个参数来控制 dropout 的概率，一个参数为 input_keep_prob，它可以用来控制输入的 dropout 概率【注意这里定义的实际上是节点被保留的概率。如果给出的数字是0.9，那么只有10%的节点会被 dropout。】；另一个为 output_keep_prob ，它可以用来控制输出的 dropout 概率。
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)

# 在使用了 dropout 的基础上定义
stacked_lstm = rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)

# 和8.3.1小节中deepRNN样例程序类似，运行前向传播过程。



