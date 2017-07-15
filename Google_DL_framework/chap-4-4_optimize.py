# coding: utf-8
# chapter 4.4    optimizer further (more)

from __future__ import division
from __future__ import print_function

import tensorflow as tf 




#==========================================
#  chapter 4.4
#==========================================


#------------------------------------------
#  chap 4.4.1    learning rate
#------------------------------------------
# 通过指数衰减的方法设置梯度下降算法中的学习率，可以让模型在训练前期快速接近较优解，又可以保证模型在训练后期不会有太大的波动，从而更加接近局部最优。


# 一种更加灵活的学习率设置方法——指数衰减法。tf.train.exponential_decay函数实现了指数衰减学习率。通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定。exponential_rate函数会指数级地减小学习率，它实现了以下代码的功能：
#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
# 其中decayed_learning_rate为每一轮优化时使用的学习率，learning_rate为事先设定的初始学习率，decay_rate为衰减系数，decay_steps为衰减速度。
# tf.train.exponential_decay函数可以通过设置参数staircase选择不同的衰减方式。staircase默认值为False，这时学习率连续衰减，否则 global_step/decay_steps会被转化成整数，这使得学习率成为一个阶梯函数（staircase function）。decay_steps通常代表了完整使用一遍训练数据所需要的迭代轮数，这个迭代轮数也就是总训练样本数除以每一个batch中的训练样本数，这种设置的常用场景是没完整过完一遍训练数据，学习率就减小一次，这可以使得训练数据集中的所有数据对模型训练有相等的作用。当使用连续的指数衰减学习率时，不同的训练数据有不同的学习率，而当学习率减小时，对应的训练数据对模型训练结果的影响也就小了。

'''
tf.train.exponential_decay

exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
'''


'''
global_step = tf.Variable(0)

# 通过exponential_decay函数生成学习率。
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

# 使用指数衰减的学习率。在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率也得到相应更新。
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(...my loss..., global_step=global_step)
'''
# 设定初始学习率为0.1，每训练100轮后学习率乘以0.96. 一般来说初始学习率、衰减系数和衰减速度都是根据经验设置的，而且损失函数下降的速度和迭代结束之后总损失的大小没有必然的联系。也就是说并不能通过前几轮损失函数下降的速度来比较不同神经网络的效果。




#------------------------------------------
#  chap 4.4.2    overfitting
#------------------------------------------

# 所谓过拟合，指的是当一个模型过于复杂之后，它可以很好地“记忆”每一个训练数据中随机噪音的部分而忘记了要去“学习”训练数据中通用的趋势。过度拟合训练数据中的随机噪声虽然可以得到非常小的损失函数，但是对于未知数据可能无法做出可靠的判断。
# 为了避免过拟合问题，一个常用方法是正则化（regularization）.正则化的思想就是在损失函数中加入刻画模型复杂程度的指标。即优化 J(theta)+lambda*R(w), J(theta)刻画模型在训练数据上的表现的损失函数，R(w)刻画模型的复杂程度，lambda表示模型复杂损失在总损失中的比例。注意这里theta表示的是一个神经网络中所有参数，包括边上的权重w和偏置项b。一般来说模型复杂度只由权重w决定。
# 常用的刻画模型复杂度的函数R(w)有两种，一种是L1正则化，另一种是L2正则化。无论是哪一种正则化方式，基本思想都是希望通过限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，但这两种也有很大区别。首先，L1正则化会让参数变得更稀疏，L2则不会，参数变得更稀疏指的是会有更多参数变为0，这样可以达到类似特征选取的功能；之所以L2不会是因为当参数很小时，其平方基本就可以忽略了，故模型不会进一步将这个参数调整为0. 其次，L1的计算公式不可导，而L2可导。因为在优化时需要计算损失函数的偏导数，所以对含有L2正则化损失函数的优化要更加简洁。优化带L1正则化的损失函数要更加复杂，而且优化方法也有很多种。在实践中也可以将L1L2同时使用 R(w)=\sum_i{ alpha*|w_i| + (1-alpha)*w_i^2 }


# 一个简单的带L2正则化的损失函数定义：
'''
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)  #tf.contrib.layers.l1_regularizer
'''
'''
tf.contrib.layers.l2_regularizer
l2_regularizer(
    scale,
    scope=None
)
'''
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
	# 输出为 (|1|+|-2|+|-3|+|4|)*0.5 = 5. 其中0.5为正则化项的权重。
	print(sess.run( tf.contrib.layers.l1_regularizer(.5)(weights) ))
	# 输出为 (1^2+(-2)^2+(-3)^2+4^2)/2 * 0.5 = 7.5 。
	print(sess.run( tf.contrib.layers.l2_regularizer(.5)(weights) ))
# TensorFlow会将L2的正则化损失值除以2使得求导得到的结果更加简洁。




# 在简单的NN中，这样就可以很好计算带正则化的损失函数了。但当NN的参数增多之后，这样方式受限可能导致损失函数loss的定义很长，可读性差且容易出错。但更主要的是，当网格结构复杂之后定义网络结构的部分和计算损失函数的部分可能不在同一个函数中，这样通过变量这种方式计算损失函数就不方便了。为了解决这个问题，可以使用TensorFlow中提供的集合（collection），它可以在一个计算图（tf.Graph）中保存一组实体（比如张量）。以下代码给出了通过集合计算一个5层NN带L2正则化的损失函数的计算方法。
# 通过使用集合的方法在网络结构比较复杂的情况下可以使代码的可读性更高。在更加复杂的网络结构中，是用这样的方式来计算损失函数将大大增强代码的可读性。

'''
# import tensorflow as tf

# 获取一层NN边上的权重，并将这个权重的L2正则化损失加入名称为‘losses’的集合中
def get_weight(shape, _lambda):  #lambda
	# 生成一个变量
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	# add_to_collection函数将这个新生成变量的L2正则化损失加入集合
	# 这个函数的第一个参数‘losses’是集合的名字，第二个参数是要加入这个集合的内容
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(_lambda)(var))
	# 返回生成的变量
	return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]  ## 连上输入层，一共是5层，第一层是输入层
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层。
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的NN结构
for i in range(1, n_layers):
	# layer_dimension[i] 为下一层的节点个数
	out_dimension = layer_dimension[i]
	# 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合。
	weight = get_weight([in_dimension, out_dimension], 0.001)
	bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))  #shape=(1,out_dimension)
	# 使用 ReLU 激活函数
	cur_layer = tf.nn.relu( tf.matmul(cur_layer, weight) + bias )
	# 进入下一层之前将下一层的节点个数更新为当前层节点个数
	in_dimension = layer_dimension[i]

# 在定义NN前向传播的同时已经将所有的L2正则化损失加入了图上的集合，这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection返回一个列表，这个列表是所有这个集合中的元素。在这个样例中，这些元素就是损失函数的不同部分，将它们加起来就可以得到最终的损失函数。
loss = tf.add_n(tf.get_collection('loss'))
'''




#------------------------------------------
#  chap 4.4.3    滑动平均模型
#------------------------------------------

# 可以使模型在测试数据上更健壮（robust）的方法——滑动平均模型。在采用随机梯度下降算法训练神经网络时，使用滑动平均模型在很多应用中都可以在一定程度提高最终模型在测试数据上的表现。
# 在TensorFlow中提供了 tf.train.ExponentialMovingAverage 来实现滑动平均模型。在初始化EMA时，需要提供一个衰减率（decay），它将用于控制模型更新的速度。EMA对每一个变量会维护一个影子变量（shadow_variable），这个影子变量的初始值就是相应变量的初始值，而每次运行变量更新时，影子变量的值会更新为 shadow_variable = decay*shadow_variable + (1-decay)*variable。其中shadow是影子变量，variable是待更新的变量，decay为衰减率。
# decay决定了模型更新的速度，decay越大则模型越趋于稳定。在实际应用中，decay一般会设成非常接近于1的数（比如0.999或0.9999）。为了使得模型在训练前期可以更新得更快，EMA还提供了num_updates参数来动态设置decay的大小。如果在EMA初始化时提供了num_updates参数，那么每次使用的衰减率将是  min{ decay, \frac{ 1+num_updates }{ 10+num_updates } }

# import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量的初始值为0. 注意这里手动指定了变量的类型为tf.float32，因为所有需要计算滑动平均的变量必须是实数型。
v1 = tf.Variable(0, dtype=tf.float32)  #0.0
# 这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率。
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类（class）。初始化时给定了衰减率（0.99）和控制衰减率的变量step。
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时这个列表中的变量都会被更新。
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
	# 初始化所有变量
	init_op = tf.global_variables_initializer()  #operation
	sess.run(init_op)

	# 通过ema.average(v1)获取滑动平均之后变量的取值。在初始化之后变量v1的值和v1的滑动平均都为0
	print(sess.run([v1, ema.average(v1)]))

	# 更新变量v1的值到5
	sess.run(tf.assign(v1, 5))
	# 更新v1的滑动平均值。衰减率为 min{0.99, (1+step)/(10+step)=0.1} = 0.1，所以v1的滑动平均会被更新为0.1*0+0.9*5=4.5
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)] ))

	# 更新step的值为10000
	sess.run(tf.assign(step, int(1e4)))  #1e4,10000
	# 更新v1的值为10
	sess.run(tf.assign(v1, 10))
	# 更新v1的滑动平均值。衰减率为 min{0.99, (1+step)/(10+step) ~~0.999} = 0.99，所以v1的滑动平均会被更新为 0.99*4.5+0.01*10=4.555
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)] ))
	# 输出 [10.0, 4.5549998]
	
	# 再次更新滑动平均值，得到的新滑动平均值为 0.99*4.555+0.01*10=4.60945
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)] ))
	# 输出 [10.0, 4.6094499]

'''
[0.0, 0.0]
[5.0, 4.5]
[10.0, 4.5549998]
[10.0, 4.6094499]
[Finished in 4.2s]

v1			0.0		
decay 		0.99	min{0.99,0.1}=0.1	min{0.1,0.1}=0.1
shadow_v1	0.0		0.99*0+0.01*0=0.0	0.1*0+0.9*0 =0.0			ema.average?
step 		0		
(1+)/(10+)	0.1		

v1 			5.0		1.4th: 10.0
decay 		0.99	1.1st: min{0.99,0.1}=0.1		2.1st: min{0.99,0.999} =0.99		3.1st: 0.99
shadow_v1 	0.0		1.2nd: 0.1*0+0.9*5  =4.5		2.2nd: 0.99*4.5+0.01*10=4.555		3.2nd: 0.99*4.555+0.01*10= 4.609449999999999
step 		0		1.3rd: 10000
(1+)/(10+) 	0.1		1.3th: 10001/10010=0.999100899100899 ~=0.999

'''




#------------------------------------------
# chap 4.4.3   again
#------------------------------------------

# import tensorflow as tf

# 获取一层NN边上的权重，并将这个权重的L2正则化损失加入名称为‘losses’的集合中
def get_weight(shape, _lambda):  #lambda
	# 生成一个变量
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	# add_to_collection函数将这个新生成变量的L2正则化损失加入集合
	# 这个函数的第一个参数‘losses’是集合的名字，第二个参数是要加入这个集合的内容
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(_lambda)(var))
	# 返回生成的变量
	return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]  ## 连上输入层，一共是5层，第一层是输入层
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层。
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的NN结构
for i in range(1, n_layers):
	# layer_dimension[i] 为下一层的节点个数
	out_dimension = layer_dimension[i]
	# 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合。
	weight = get_weight([in_dimension, out_dimension], 0.001)
	bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))  #shape=(1,out_dimension)
	# 使用 ReLU 激活函数
	cur_layer = tf.nn.relu( tf.matmul(cur_layer, weight) + bias )
	# 进入下一层之前将下一层的节点个数更新为当前层节点个数
	in_dimension = layer_dimension[i]

# 在定义NN前向传播的同时已经将所有的L2正则化损失加入了图上的集合，这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection返回一个列表，这个列表是所有这个集合中的元素。在这个样例中，这些元素就是损失函数的不同部分，将它们加起来就可以得到最终的损失函数。
loss = tf.add_n(tf.get_collection('loss'))




