# -*- coding: utf-8 -*-
#  chapter 8.4.2  时间序列预测
#    part 1  使用 TFLearn 自定义模型
#    part 2  预测正弦函数

'''
References:

Question 1:
    output, _ = tf.nn.rnn.rnn(cell, x_, dtype=tf.float32)
AttributeError: 'module' object has no attribute 'rnn'
Answer 1:
https://stackoverflow.com/questions/42552540/tensorflow-tf-contrib-rnn-module-object-is-not-callable

Question 2:
ValueError: Trying to share variable rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel, but specified shape (60, 120) and found shape (40, 120).
Answer 2:
https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k

Question 3:
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_steps(), optimizer="Adagrad", learning_rate=0.1)
AttributeError: 'module' object has no attribute 'get_global_steps'
Answer 3:
https://github.com/songgc/TF-recomm/issues/3
https://www.tensorflow.org/api_docs/python/tf/contrib/framework/get_global_step
https://www.tensorflow.org/api_docs/python/tf/contrib/framework/get_or_create_global_step

Answer 4:
https://stackoverflow.com/questions/11983024/matplotlib-legends-not-working

'''




from __future__ import print_function

import numpy as np 
import tensorflow as tf 

from tensorflow.python.ops import rnn, rnn_cell

# 加载 matplotlib 工具包，使用该工具可以对预测的 sin 函数曲线进行绘图。
import matplotlib as mpl 
mpl.use('Agg')
from matplotlib import pyplot as plt 


learn = tf.contrib.learn

HIDDEN_SIZE = 30 	# LSTM 中隐藏节点的个数。
NUM_LAYERS = 2 		# LSTM 的层数。

TIMESTEPS = 10 			# 循环神经网络的截断长度。
TRAINING_STEPS = 10000 	# 训练轮数。
BATCH_SIZE = 32			# batch 大小。

TRAINING_EXAMPLES = 10000 	# 训练数据个数。
TESTING_EXAMPLES = 1000 	# 测试数据个数。
SAMPLE_GAP = 0.01 			# 采样间隔。


def generate_data(seq):
	X = []
	y = []
	# 序列的第 i 项和后面的 TIMESTEPS-1 项合在一起作为输入；第 i+TIMESTEPS 项作为输出。即用 sin 函数前面的 TIMESTEPS 个点的信息，预测第 i+TIMESTEPS 个点的函数值。
	for i in range(len(seq) - TIMESTEPS -1):
		X.append([seq[i : i+TIMESTEPS]])
		y.append([seq[i + TIMESTEPS]])
	return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y):
	# 使用多层的 lstm 结构。
	##lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
	##cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
	##lstm_cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
	##cell = rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
	def lstm_cell():
		return tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
	cell = tf.contrib.rnn.MultiRNNCell( [lstm_cell()  for _ in range(NUM_LAYERS)] )  ## stacked_lstm
	## x_ = tf.unpack(X, axis=1)
	x_ = tf.unstack(X, axis=1)

	# 使用 TensorFlow 接口将多层的 LSTM 结构连接成 RNN 网络并计算其前向传播结果。
	##output, _ = tf.nn.rnn(cell, x_, dtype=tf.float32)
	output, _ = rnn.static_rnn(cell, x_, dtype=tf.float32)
	# 在本问题中只关注最后一个时刻的输出结果，该结果为下一时刻的预测值。
	output = output[-1]

	# 对 LSTM 网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均平方差损失函数。
	prediction, loss = learn.models.linear_regression(output, y)

	# 创建模型优化器并得到优化步骤。
	train_op = tf.contrib.layers.optimize_loss(
		loss, 
		tf.contrib.framework.get_global_step(), ##tf.contrib.framework.get_global_steps(), 
		optimizer="Adagrad", 
		learning_rate=0.1)

	return prediction, loss, train_op


# 建立深层循环网络模型。
regressor = learn.Estimator(model_fn = lstm_model)

# 用正弦函数生成训练和测试数据集合。
# numpy.linspace 函数可以创建一个等差序列的数组，它常用的参数有三个参数，第一个参数表示起始值，第二个参数表示终止值，第三个参数表示数列的长度。例如，linespace(1,10,10) 产生的数组是 array([1,2,3,4,5,6,7,8,9,10]) 。
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32) ))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32) ))

# 调用 fit 函数训练模型。
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 使用训练好的模型对测试数据进行预测。
predicted = [[pred]  for pred in regressor.predict(test_X)]
# 计算 rmse 作为评价指标。
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print ("Mean Square Error is: %f" % rmse[0])

'''
运行以上程序可以得到输出：
从输出可以看出通过循环神经网络可以非常精确地预测正弦函数 sin 的取值。
'''

# 对预测的 sin 函数曲线进行绘图，并存储到运行目录下的 sin.png
fig = plt.figure()
##plot_predicted = plt.plot(predicted, label='predicted')
##plot_test = plt.plot(test_y, label='real_sin')
##plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
plot_test = plt.plot(test_y, label='real_sin')
plot_predicted = plt.plot(predicted, label='predicted')
plt.legend()
# 得到的结果如 图8-21 所示。
fig.savefig('chap-8.4.2_sin.png')





'''
References:
https://github.com/tensorflow/tensorflow/issues/7550

tf.unpack（A, axis）是一个解包函数。A是一个需要被解包的对象，axis是一个解包方式的定义，默认是零，如果是零，返回的结果就是按行解包。如果是1，就是按列解包。

As far as I know, tf.pack has been renamed as tf.stack.

'''




'''
Mean Square Error is: 0.002101
[Finished in 19.4s]
Mean Square Error is: 0.001643
[Finished in 18.9s]






WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpVEcgcm
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:109: calling fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with y is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:109: calling fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:109: calling fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with batch_size is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/models.py:107: mean_squared_error_regressor (from tensorflow.contrib.learn.python.learn.ops.losses_ops) is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Use `tf.contrib.losses.mean_squared_error` and explicit logits computation.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/ops/losses_ops.py:39: mean_squared_error (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.mean_squared_error instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:539: compute_weighted_loss (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.compute_weighted_loss instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:151: add_loss (from tensorflow.contrib.framework.python.ops.arg_scope) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.add_loss instead.
2017-07-25 16:04:46.558956: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-25 16:04:46.560853: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-25 16:04:46.566146: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:112: calling predict (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/models.py:107: mean_squared_error_regressor (from tensorflow.contrib.learn.python.learn.ops.losses_ops) is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Use `tf.contrib.losses.mean_squared_error` and explicit logits computation.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/ops/losses_ops.py:39: mean_squared_error (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.mean_squared_error instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:539: compute_weighted_loss (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.compute_weighted_loss instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:151: add_loss (from tensorflow.contrib.framework.python.ops.arg_scope) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.add_loss instead.
Mean Square Error is: 0.002229
/usr/local/lib/python2.7/dist-packages/matplotlib/legend.py:634: UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x7fa56eaea550>] instances.
A proxy artist may be used instead.
See: http://matplotlib.org/users/legend_guide.html#using-proxy-artist
  "#using-proxy-artist".format(orig_handle)
/usr/local/lib/python2.7/dist-packages/matplotlib/legend.py:634: UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x7fa56eaea750>] instances.
A proxy artist may be used instead.
See: http://matplotlib.org/users/legend_guide.html#using-proxy-artist
  "#using-proxy-artist".format(orig_handle)
[Finished in 18.9s]
'''




