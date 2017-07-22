# coding: utf-8
#  chapter 5.3    at last
#      modify "inference" on the basis of 5.2.1




#------------------------------------
#  basic:  chap 5.2.1   train NN
#------------------------------------

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784		# 输入层的节点数。
OUTPUT_NODE = 10		# 输出层的节点数。


LAYER1_NODE = 50 #500		# 隐藏层节点数。
BATCH_SIZE = 100			# 一个训练batch中的训练数据个数。
LEARNING_RATE_BASE = 0.8			# 基础的学习率。
LEARNING_RATE_DECAY = 0.99			# 学习的衰减率。
REGUALRIZATION_RATE = 0.0001		# 描述模型复杂度的正则化项在损失函数中的系数。
TRAINING_STEPS = 100#30000			# 训练轮数。
MOVING_AVERAGE_DECAY = 0.99			# 滑动平均衰减率。




'''
# 一个辅助函数，给定NN的输入和所有参数，计算NN的前向传播结果。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	# 当没有提供滑动平均类时，直接使用参数当前的取值。
	if avg_class == None:
		# 计算隐藏层的前向传播结果，这里使用了ReLU激活函数。
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

		# 计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因为预测时使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果的计算没有影响。于是在计算整个神经网络的前向传播时可以不加入最后的softmax层。
		return tf.matmul(layer1, weights2 ) + biases2

	else:
		# 首先使用 avg_class.average 函数来计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果。
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(biases2))
'''




def inference(input_tensor, avg_class, reuse=False):
	if avg_class == None:
		# 定义第一层神经网络的变量和前向传播过程。
		with tf.variable_scope('layer1', reuse=reuse):
			# 根据传进来的reuse来判断是创建新变量还是使用已经创建好的。在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传进来了。
			weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
			biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
			layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

		# 类似地定义第二层神经网络的变量和前向传播过程。
		with tf.variable_scope("layer2", reuse=reuse):
			weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
			biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
			layer2 = tf.matmul(layer1, weights) + biases
	
		# 返回最后的前向传播结果。
		return layer2
	
	else:
		with tf.variable_scope('layer1', reuse=reuse):
			weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
			biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
			layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))
		with tf.variable_scope("layer2", reuse=reuse):
			weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
			biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
			layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)
		return layer2

'''
def inference(input_tensor, reuse=False):
	# 定义第一层神经网络的变量和前向传播过程。
	with tf.variable_scope('layer1', reuse=reuse):
		# 根据传进来的reuse来判断是创建新变量还是使用已经创建好的。在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传进来了。
		weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
		biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

	# 类似地定义第二层神经网络的变量和前向传播过程。
	with tf.variable_scope("layer2", reuse=reuse):
		weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
		biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases
	
	# 返回最后的前向传播结果。
	return layer2

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y = tf.inference(x)
# 在程序中需要使用训练好的NN进行推导时，可以直接调用 inference(new_x, True) 。如果需要使用滑动平均模型可以参考5.2.1小节中使用的代码，把计算滑动平均的类传到 inference 函数中即可。获取或者创建变量的部分不需要改变。
new_x = ...
new_y = inference(new_x, True)


使用上面这段代码所示的方式，就不再需要将所有变量都作为参数传递到不同的函数中了。当NN结构更加复杂、参数更多时，使用这种变量管理的方式将大大提高程序的可读性。
'''




# 训练模型的过程。
def train(mnist):
	x = tf.placeholder(tf.float32, [None,INPUT_NODE], name='x-input')
	y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name='y-input')

	'''
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE] ))
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE], stddev=0.1 ))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE] ))
	y = inference(x, None, weights1, biases1, weights2, biases2)
	'''
	y = inference(x, None, False)

	global_step = tf.Variable(0, trainable=False)
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	'''
	average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
	'''
	average_y = inference(x, variable_averages, True)

	'''
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, axis=1), logits=y)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	regularizer = tf.contrib.layers.l2_regularizer(REGUALRIZATION_RATE)  ##this's lambda
	regularization = regularizer(weights1) + regularizer(weights2)
	loss = cross_entropy_mean + regularization
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,						# 基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减。
		global_step,							# 当前迭代的轮数。
		mnist.train.num_examples / BATCH_SIZE,	# 过完所有的训练数据需要的迭代次数。
		LEARNING_RATE_DECAY,					# 学习率衰减速度。
		staircase=True)  ##not before

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	with tf.control_dependencies([train_step, variables_averages_op]):
		train_op = tf.no_op(name='train')

	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	'''
	with tf.variable_scope("", reuse=True):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, axis=1), logits=y)
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		regularizer = tf.contrib.layers.l2_regularizer(REGUALRIZATION_RATE)  ##this's lambda
		regularization = regularizer(tf.get_variable("layer1/weights")) + regularizer(tf.get_variable("layer2/weights"))
		loss = cross_entropy_mean + regularization
		learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True) 
		regularization = regularizer( tf.get_variable("layer1/weights") ) + regularizer(tf.get_variable("layer2/weights"))
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

		with tf.control_dependencies([train_step, variables_averages_op]):
			train_op = tf.no_op(name='train')

		correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		test_feed = {x: mnist.test.images, y_: mnist.test.labels}

		for i in range(TRAINING_STEPS):
			if i % 10 == 0:  ##1000
				validate_acc = sess.run(accuracy, feed_dict=validate_feed)
				print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))  ##指数(e)或浮点数 (根据显示长度)

			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict={x:xs, y_:ys})

		test_acc = sess.run(accuracy, feed_dict=test_feed)
		print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


# 主程序入口。
def main(argv=None):
	mnist = input_data.read_data_sets("/home/ubuntu/Program/GoogleDL_datasets/MNIST_data/", one_hot=True)
	train(mnist)

# TensorFlow提供的一个主程序入口，tf.app.run 会调用上面定义的main函数。
if __name__ == '__main__':
	tf.app.run()




#------------------------------------
#------------------------------------
'''
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/train-images-idx3-ubyte.gz
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/t10k-labels-idx1-ubyte.gz
2017-07-22 23:35:41.018566: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-22 23:35:41.018693: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-22 23:35:41.018711: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
After 0 training step(s), validation accuracy using average model is 0.0606 
After 10 training step(s), validation accuracy using average model is 0.7098 
After 20 training step(s), validation accuracy using average model is 0.7878 
After 30 training step(s), validation accuracy using average model is 0.826 
After 40 training step(s), validation accuracy using average model is 0.8574 
After 50 training step(s), validation accuracy using average model is 0.8812 
After 60 training step(s), validation accuracy using average model is 0.895 
After 70 training step(s), validation accuracy using average model is 0.9004 
After 80 training step(s), validation accuracy using average model is 0.904 
After 90 training step(s), validation accuracy using average model is 0.9042 
After 100 training step(s), test accuracy using average model is 0.9132
[Finished in 4.7s]


这次算是优化？比5.2.1的结果好多了貌似
'''



