# coding: utf-8
# Google DeepLearning framework: TensorFlow
# chapter 3.2   tensor

from __future__ import print_function




#=============================
#  chap 3.2.1  concept
#=============================

import tensorflow as tf 

# tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中。
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="add")
print(result)
# Output:  Tensor("add:0", shape=(2,), dtype=float32)

'''
# 类型检查：类型不匹配错误
a = tf.constant([1, 2], name="a")
b = tf.constant([2., 3.], name="b")
result = a + b
# ValueError: Tensor conversion requested dtype int32 for Tensor with dtype float32: 'Tensor("b_1:0", shape=(2,), dtype=float32)'
'''



#==============================
#  chap 3.2.2  use tensor
#==============================

# 第一类用途是对中间计算结果的引用

# 使用张量记录中间结果
a = tf.constant([1., 2.], name="a")
b = tf.constant([2., 3.], name="b")
result = a + b
# 直接计算向量的和，这样可读性会比较差。
result = tf.constant([1., 2.], name="a") + tf.constant([2., 3.], name="a")

# 使用张量的第二类情况是当计算图构造完成之后，张量可以用来获得计算结果，也就是得到真实的数字。虽然张量本身没有存储具体的数字，但是通过会话，就可以得到这些具体的数字。



#======================================
#  chap 3.2.3  running model: session
#======================================

# 实用会话来执行定义好的计算。会话拥有并管理TensorFlow程序运行时的所有资源。当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄露的问题。
# 使用会话的模式一般有两种。第一种模式需要明确调用会话生成函数和关闭会话函数。第二种可以通过Python的上下文管理器来使用会话。


# 创建一个会话
sess = tf.Session()
# 使用这个创建好的会话来得到关心的运算的结果。比如可以调用sess.run(result)，来得到3.1节样例中张量result的取值。
print(sess.run(result))
# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close()

# 创建一个会话，并通过Python中的上下文管理器来管理这个会话。
with tf.Session() as sess:
	# 使用这创建好的会话来计算关心的结果。
	print(sess.run(result))
# 不需要再调用“Session.close()”函数来关闭会话，当上下文退出时会话关闭和资源释放也自动完成了。


# TensorFlow不会自动生成默认的会话，而是需要手动指定。当默认的会话被指定之后可以通过tf.Tensor.eval函数来计算一个张量的取值。以下代码展示了通过设定默认会话计算张量的取值。
sess = tf.Session()
with sess.as_default():
	print(result.eval())
sess.close()
# 以下代码也可以完成相同功能。

sess = tf.Session()
# 下面的两个命令有相同的功能
print(sess.run(result))
print(result.eval(session=sess))
sess.close()


# 在交互式环境下直接构建默认会话的函数。使用这个函数会自动将生成的会话注册为默认会话。可以省去将产生的会话注册为默认会话的过程。
sess = tf.InteractiveSession()
print(result.eval())
sess.close()


# 两种都可以通过 ConfigProto Protocol Buffer 来配置需要生成的会话。下面给出了通过 ConfigProto 配置会话的方法。
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
# 通过它可以配置类似并行的线程数、GPU分配策略、运算超时时间等参数。最常使用的有两个。
# 其一是allow_soft_placement，默认为False，设为True可以当某些运算无法被当前GPU支持时可以自动调整到CPU上而不是报错。类似，通过将之设为True可以让程序在拥有不同数量的GPU机器上顺利运行。
# 其二是log_device_placement，当其为True时，日志中将会记录每个节点被安排在了哪个设备上以方便调试。而在生产环境中将这个参数设置为False可以减少日志量。
sess1.close()
sess2.close()


