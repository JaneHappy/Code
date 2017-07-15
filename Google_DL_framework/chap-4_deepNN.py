# coding: utf-8
# chapter 4.1 ~ 4.3

from __future__ import print_function
import tensorflow as tf
import numpy as np




#==========================================
#  chap 4.1    activation (function)
#==========================================

# chap 4.1.2
# 目前TensorFlow提供了7种不同的非线性激活函数，tf.nn.relu、tf.sigmoid、和tf.tanh是其中比较常用的几个。当然，TensorFlow也支持使用自己定义的激活函数。

'''
a = tf.nn.relu(tf.matmul(x,w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)
'''


# chap 4.1.3
# 感知机无法解决异或问题。加入隐藏层可解决。深层神经网络实际上有组合特征提取的功能，这个特性对于解决不易提取特征向量的问题（比如图片识别、语音识别等）有很大帮助。这也是深度学习在这些问题上更加容易取得突破性进展的原因。




#==========================================
#  chap 4.2    loss function
#==========================================

# chap 4.2.1
# 交叉熵非对称，刻画的是通过概率分布q来表达概率分布p的困难程度。p代表正确答案，是希望得到的结果，q代表预测值。交叉熵刻画的是两个概率分布的距离，也就是说交叉熵值越小，两个概率分布越接近。

# cross_entropy = -tf.reduce_mean( y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) )
# cross_entropy = -tf.reduce_mean(tf.multiply( y_, tf.log(tf.clip_by_value(y, 1e-10, 1.0)) ))

# 通过tf.clip_by_value函数就可以保证在进行log（ln(x)）运算时，不会出现log0这样的错误或者大于1的概率。

v = tf.constant([[1.,2,3], [4,5,6]])
sess = tf.Session()
with sess.as_default():
	print(tf.clip_by_value(v, 2.5, 4.5).eval())
	'''
	[[ 2.5  2.5  3. ]
	 [ 4.   4.5  4.5]]
	'''
sess.close()

v = tf.constant([1, 2, 3, np.e ])
with tf.Session() as sess:
	print(tf.log(v).eval())

v1 = tf.constant([[1.,2], [3,4]])
v2 = tf.constant([[5.,6], [7,8]])
with tf.Session() as sess:
	print((v1 * v2).eval())
	print(tf.multiply(v1, v2).eval())
	print(tf.matmul(v1, v2).eval())

v = tf.constant([[1.,2,3], [4,5,6]])
with tf.Session() as sess:
	print(tf.reduce_mean(v).eval())
	print(tf.reduce_sum(v).eval())


# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_) ##y_, Real label


# 回归问题解决的是对具体数值的预测，比如房价预测、销量预测等。这些问题需要预测的不是一个事先定义好的类别，而是一个任意实数。解决回归问题的神经网络一般只有一个输出节点，其输出值就是预测值。其最常用的损失函数是均方误差。
# 均方误差 MSE，mean squared error   = \frac{1}{n} * \sum()^2
# mse = tf.reduce_mean(tf.square(y_ - y))
# mse = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
# 均方误差也是分类问题中常用的一种损失函数。y代表输出答案，y_代表标准答案。




# chap 4.2.2  自定义损失函数
# TensorFlow不仅支持经典的损失函数，还可以优化任意的自定义损失函数。本小节介绍如何通过自定义损失函数的方法，使得神经网络优化的结果更加接近实际问题的需求。


'''
Loss(y,y') = \sum_{i=1}^n{f(y_i, y'_i)}
f(x,y) = a(x-y)  if x>y  or  b(y-x)
'''
# loss = tf.reduce_sum(tf.select(tf.greater(v1,v2), (v1-v2)*a, (v2-v1)*b))


v1 = tf.constant([1., 2, 3, 4])
v2 = tf.constant([4., 3, 2, 1])
v3 = tf.constant([-1., 3, 0, 2.5, 7])
sess = tf.InteractiveSession()
print(tf.greater(v1, v2).eval())  #[False False  True  True]
print(tf.where(tf.greater(v1,v2), v1, v2).eval())  #[ 4.  3.  3.  4.]  #print(tf.select(tf.greater(v1,v2), v1, v2).eval())
'''
print(tf.greater(v3, v1).eval())  print(tf.greater(v1, v3).eval())
ValueError: Dimensions must be equal, but are 4 and 5 for 'Greater_2' (op: 'Greater') with input shapes: [4], [5].
'''
sess.close()



# import tensorflow as tf
# from numpy.random import RandomState

batch_size = 8

# 两个输入节点。
x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
# 回归问题一般只有一个输出节点。
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

# 定义了一个单层的神经网络前向传播过程，这里就是简单加权和。
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本。
loss_less = 10.  #损失利润
loss_more = 1.   #损失成本
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
							  (y - y_) * loss_more,
							  (y_ - y) * loss_less ))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 通过随机数生成一个模拟数据集。
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归的正确值为两个输入的和加上一个随机量。之所以要加上一个随机量是为了加入不可预测的噪音，否则不同损失函数的意义就不大了，因为不同损失函数都会在能完全预测正确的时候最低。一般来说噪音为一个均值为0的小量，所以这里的噪音设置为 -0.05 ~ 0.05 的随机数。
Y = [[x1 + x2 + rdm.rand()/10.0-0.05]  for (x1,x2) in X]
# rdm.rand()  \in [0,1], (0,1)?  [0,1)

# 训练神经网络
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print("Before training, w =", sess.run(w1))
	STEPS = 50 #5000
	for i in range(STEPS):
		start = (i * batch_size) % dataset_size
		end = min(start+batch_size, dataset_size)
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end] })
		#print("After  training, w =", sess.run(w1))
	print("After  training, w =", sess.run(w1))




#==========================================
#  chap 4.3    Optimization algorithms
#  a)  backpropagation  反向传播算法
#  b)  gradient decent  梯度下降算法
#==========================================

# 学习率 eta (learning rate) 定义的是每次参数更新的幅度。直观理解为每次参数移动的幅度。

# 梯度下降算法并不能保证被优化的函数达到全局最优解。在训练神经网络时，参数的初始值会很大程度影响最后得到的结果。只有当损失函数为凸函数时，梯度下降算法才能保证达到全局最优解。
# 除了不一定能达到全局最优外，梯度下降算法的另外一个问题就是计算时间太长。因为在每一轮迭代中都需要计算在全部训练数据上的损失函数。为加速训练过程，可使用随机梯度下降算法（stochastic gradient descent），它优化的不是在全部训练数据上的损失函数，而是在每一轮迭代中，随机优化某一条训练数据上的损失函数，这样每一轮参数更新的速度就大大加快了。但是问题也很明显：在某一条数据上损失函数更小并不代表在全部数据上损失函数更小，于是使用随机梯度下降优化得到的神经网络甚至可能无法达到局部最优。
# 为综合两者优点，在实际应用中一般采用这两个算法的折中——每次计算一小部分训练数据的损失函数，这一小部分数据被称之为一个batch。通过矩阵运算，每次在一个batch上优化神经网络的参数并不会比单个数据慢太多；另一方面，每次使用一个batch可以大大减小收敛所需要的迭代次数，同时可以使收敛到的结果更加接近梯度下降的效果。


'''
batch_size = n

# 每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# 定义神经网络结构和优化算法
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 训练神经网络。
with tf.Session() as sess:
	# 参数初始化。
	...
	# 迭代地更新参数。
	for i in range(STEPS):
		# 准备batch_size个训练数据。一般将所有训练数据随机打乱之后再选取可以得到更好的优化效果。
		current_X, current_Y = ...
		sess.run(train_step, feed_dict={x: current_X, y_: current_Y})
'''




