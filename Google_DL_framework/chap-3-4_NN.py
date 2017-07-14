# coding: utf-8
# Neural Network

from __future__ import print_function




#==========================================
#  chap 3.4.1    function, procedure
#==========================================

# http://playground.tensorflow.org




#==========================================
#  chap 3.4.2    forward-propagation
#==========================================

# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)



#==========================================
#  chap 3.4.3    parameters
#==========================================

'''

# 在神经网络中，给参数赋予随机初始值最为常见，所以一般也使用随机数给TensorFlow中的变量初始化。
weights = tf.Variable(tf.random_normal([2,3], stddev=2.))
# tf.truncated_normal,  tf.random_uniform,  tf.random_gamma

# 在神经网络中，偏置项（bias）通常会使用常数来设置初始值。
biases = tf.Variable(tf.zeros([3])) # ?(tf.)float32
# tf.ones,  tf.fill([2,3],9) ->[[9,9,9],[9,9,9]],  tf.constant([1,2,3])

# 除了使用随机数或者常数，TensorFlow也支持通过其他变量的初始值来初始化新的变量。以下代码给出了具体的方法。
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value() * 2.0)

'''


'''
import tensorflow as tf

# 声明 w1,w2 两个变量。这里还通过seed参数设定了随机种子，这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里x是一个1*2的矩阵。
x = tf.constant([[0.7, 0.9]])

# 通过3.4.2小节描述的前向传播算法获得神经网络的输出。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 与3.4.2中的计算不同，这里不能直接通过sess.run(y)来获取y的取值，因为w1和w2都还没有运行初始化过程。下面的两行分别初始化了w1和w2两个变量。
#sess.run(w1.initializer)  # 初始化w1
#sess.run(w2.initializer)  # 初始化w2
# 输出[[3.95757794]]
#print(sess.run(y))
#sess.close()


# 更便捷的方式来完成变量初始化过程。通过tf.initialize_all_variables函数实现初始化所有变量的过程。
init_op = tf.initialize_all_variables()
print(sess.run(init_op))
print(sess.run(y), '\n',sess.run(w1),'\n',sess.run(w2),'\n',sess.run(a))
sess.close()
# 不需要将变量一个一个初始化了，这个函数也会自动处理变量之间的依赖关系。
'''




#==========================================
#  chap 3.4.4    back-propagation
#==========================================

'''

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3], stddev=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1))

# 定义placeholder作为存放输入数据的地方。这里唯独也不一定要定义。但如果维度是确定的，那么给出维度可以降低出错的概率。
x = tf.placeholder(tf.float32, shape=(3,2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
#init_op = tf.initialize_all_variables()  ## Instructions for updating: Use `tf.global_variables_initializer` instead.
sess.run(init_op)
print(sess.run(y, feed_dict={x: [[0.7,0.9], [0.1,0.4], [0.5,0.8]] }))

'''


# 定义损失函数来刻画预测值与真实值的差距。
#cross_entropy = -tf.reduce_mean( y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) )
# 定义学习率，在第4章中将更加具体的介绍学习率。
#learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数。
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)




#==========================================
#  chap 3.4.5    a simple sample
#==========================================

import tensorflow as tf

# Numpy是一个科学计算的工具包，这里通过Numpy工具包生成模拟数据集。
from numpy.random import RandomState

# 定义训练数据batch的大小。
batch_size = 8

# 定义神经网络的参数，这里还是沿用3.4.2小节中给出的神经网络结构。
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1), trainable=True)
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1), trainable=True)

# 在shape的一个维度上使用None可以方便使用不同的batch大小。在训练时需要把数据分成比较小的batch，但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

# 定义神经网络前向传播的过程。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法。
cross_entropy = -tf.reduce_mean( y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) )  #tf.log=ln=log_e
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集。
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），而其他为负样本（比如零件不合格）。和TensorFlow游乐场中的表示法不大一样的地方是，在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法。
Y = [[int(x1+x2<1)] for (x1,x2) in X]

# 创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	# 初始化变量。
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))
	'''
	在训练之前神经网络参数的值：
	w1 = [[-0.81131822  1.48459876  0.06532937], [-2.44270396  0.0992484   0.59122431]]
	w2 = [[-0.81131822], [ 1.48459876], [ 0.06532937]]
	'''

	# 设定训练的轮数。
	STEPS = 50 #5000
	for i in range(STEPS):
		# 每次选取batch_size个样本进行训练。
		start = (i * batch_size) % dataset_size
		end = min(start+batch_size, dataset_size)

		# 通过选取的样本训练神经网络并更新参数。
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end] })
		if i % 10 == 0: #1000
			# 每隔一段时间计算在所有数据上的交叉熵并输出。
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y })
			print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))
			'''
			输出结果:
			After 0 training step(s), cross entropy on all data is 0.0674925
			After 10 training step(s), cross entropy on all data is 0.0659034
			After 20 training step(s), cross entropy on all data is 0.0643085
			After 30 training step(s), cross entropy on all data is 0.0628051
			After 40 training step(s), cross entropy on all data is 0.0614568
			通过这个结果可以发现随着训练的进行，交叉熵是逐渐变小的。交叉熵越小说明预测的结果和真实的结果差距越小。
			'''

	print(sess.run(w1))
	print(sess.run(w2))
	'''
	在训练之后神经网络参数的值：
	w1 = [[-0.84684235  1.52004194  0.10257635]. [-2.4755106   0.13197781  0.6256628 ]]
	w2 = [[-0.84565806], [ 1.52062511], [ 0.09908549]]
	可以发现这两个参数的取值已经发生了变化，这个变化就是训练的结果。它使得这个神经网络能更好地拟合提供的训练数据。
	'''



'''
2017-07-15 00:10:43.098002: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.

2017-07-15 00:10:43.098971: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-15 00:10:43.099002: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

[Finished in 2.3s]
'''



