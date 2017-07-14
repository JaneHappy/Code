# coding: utf-8
# Google DeepLearning framework
# chapter 3

from __future__ import print_function





#=============================
#  chapter 3.1    Graph
#=============================

import tensorflow as tf 


a = tf.constant([1., 2.], name="a")
b = tf.constant([2., 3.], name="b")
result = a + b

# 通过a.graph可以查看张量所属的计算图。因为没有特意指定，所以这个计算图应该等于当前默认的计算图。所以下面这个操作输出值为True。
print(a.graph is tf.get_default_graph())
## print(a.graph, b.graph, result.graph)



g = tf.Graph()
# 指定计算运行的设备
with g.device('/gpu:0'):
	result = a + b




g1 = tf.Graph()
with g1.as_default():
	# 在计算图g1中定义变量“v”，并设置初始值为0.
	v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer()) ##v = tf.get_variable("v", [0.]) #initializer=tf.zeros_initializer(shape=[1])

g2 = tf.Graph()
with g2.as_default():
	# 在计算图g2中定义变量“v”，并设置初始值为1.
	v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer()) ##v = tf.get_variable("v", [1.]) #initializer=tf.ones_initializer(shape=[1]))

# 在计算图g1中读取变量“v”的取值
with tf.Session(graph=g1) as sess:
	tf.global_variables_initializer().run() ##tf.initialize_all_variables().run()
	with tf.variable_scope("", reuse=True):
		# 在计算图g1中，变量“v”的取值应该为0，所以下面这行会输出[0.]。
		print(sess.run(tf.get_variable("v")))

# 在计算图g1中读取变量“v”的取值
with tf.Session(graph=g2) as sess:
	tf.global_variables_initializer().run() ##tf.initialize_all_variables().run()
	with tf.variable_scope("", reuse=True):
		# 在计算图g2中，变量“v”的取值应该为1，所以下面这行会输出[1.]。
		print(sess.run(tf.get_variable("v")))






#==================================
#  chap 3.1.2    make use of
#==================================


