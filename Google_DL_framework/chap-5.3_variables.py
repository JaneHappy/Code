# coding: utf-8
#  chapter 5.3  manager variables

import tensorflow as tf 




'''
# 下面这两个定义是等价的。
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")
'''





# 在名字为foo 的命名空间内创建名字为v的变量。
with tf.variable_scope("foo"):
	v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间foo中已经存在名字为v的变量，所有下面的代码将会报错：
# Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope?
##with tf.variable_scope("foo"):
##	v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable函数将直接获取已经声明的变量。
with tf.variable_scope("foo", reuse=True):
	v1 = tf.get_variable("v", [1])
	print v == v1
	# 输出为True，代表v,v1代表的是相同的TensorFlow中变量。

# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量。因为在命名空间bar中还没有创建变量v，所以下面的代码将会报错：
# Variable bar/v does not exist, disallowed. Did you mean to set reuse=None in VarScope?
##with tf.variable_scope("bar", reuse=True):
##	v = tf.get_variable("v", [1])





# 当tf.variable_scope 函数使用参数 reuse=True 生成上下文管理器时，这个上下文管理器内所有的 tf.get_variable 函数会直接获取已经创建的变量。如果变量不存在，则 tf.get_variable 函数将报错；相反，如果 tf.variable_scope 函数使用参数 reuse=None 或 reuse=False 创建上下文管理器，tf.get_variable 操作将创建新的变量。如果同名的变量已经存在，则 tf.get_variable 函数将报错。TensorFlow 中 tf.variable_scope 函数是可以嵌套的。下面的程序说明了当 tf.variable_scope 函数嵌套时，reuse 参数的取值是如何确定的。

with tf.variable_scope("root"):
	# 可以通过tf.get_variable_scope().reuse函数来获取当前上下文管理器中reuse参数的取值。
	print "root\t", tf.get_variable_scope().reuse 			# 输出False，即最外层reuse是False。

	# 新建一个嵌套的上下文管理器，并指定reuse为True。
	with tf.variable_scope("foo", reuse=True):
		print ' foo\t', tf.get_variable_scope().reuse 		# 输出True。

		# 新建一个嵌套的上下文管理器但不指定reuse，这时reuse的取值会和外面一层保持一致。
		with tf.variable_scope("bar"):
			print ' bar\t', tf.get_variable_scope().reuse 	# 输出True。

		print '-bar\t', tf.get_variable_scope().reuse
	print '-foo\t', tf.get_variable_scope().reuse  			# 输出False。退出reuse设置为True的上下文之后，reuse的值又回到了False。

'''
True
root	False
 foo	True
 bar	True
-bar	True
-foo	False
[Finished in 1.1s]
'''




# tf.variable_scope 函数生成的上下文管理器也会创建一个 TensorFlow 中的命名空间，在命名空间内创建的变量名称都会带上这个命名空间名作为前缀。所以，tf.variable_scope 函数除了可以控制 tf.get_variable 执行的功能之外，这个函数也提供了一个管理变量命名空间的方式。以下代码显示了如何通过 tf.variable_scope 来管理变量的名称。

v1 = tf.get_variable("v", [1])
print v1.name 			# 输出 v:0. "v"为变量的名称，":0"表示这个变量是生成变量这个运算的第一个结果。

with tf.variable_scope("foo"):
	v2 = tf.get_variable("v", [1])
	print v2.name 		# 输出 foo/v:0. 在 tf.variable_scope 中创建的变量，名称前面会加入命名空间的名称，并通过/来分隔命名空间的名称和变量的名称。

with tf.variable_scope("foo"):
	##v7 = tf.get_variable("v2", [1]);		print v7.name
	with tf.variable_scope("bar"):
		v3 = tf.get_variable("v", [1])
		print v3.name 	# 输出 foo/bar/v:0. 命名空间可以嵌套，同时变量的名称也会加入所有命名空间的名称作为前缀。

	v4 = tf.get_variable("v1", [1])
	print v4.name 		# 输出 foo/v1:0. 当命名空间退出之后，变量名称也就不会再被加入其前缀了。

# 创建一个名称为空的命名空间，并设置 reuse=True。
with tf.variable_scope("", reuse=True):
	v5 = tf.get_variable("foo/bar/v", [1]) 	# 可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。比如这里通过指定名称 foo/bar/v 来获取在命名空间 foo/bar/ 中创建的变量。
	print v5.name,  v5 == v3 				# 输出True。
	v6 = tf.get_variable("foo/v1", [1])
	print v6.name,  v6 == v4 				# 输出True。




