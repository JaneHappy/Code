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




