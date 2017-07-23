# coding: utf-8
#  chapter 5.4  TensorFlow 模型持久化




#----------------------------------------
# 虽然上面的程序只指定了一个文件路径，但是在这个文件目录下会出现三个文件。这是因为TensorFlow会将计算图的结构和图上参数取值分开保存。
# model.ckpt.meta 保存了TensorFlow计算图的结构，可简单理解为NN的网络结构。model.ckpt 保存了TensorFlow程序中每一个变量的取值。checkpoint 文件中保存了一个目录下所有的模型文件列表。
#----------------------------------------
'''
import tensorflow as tf 

# 声明两个变量并计算它们的和。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
# 声明 tf.train.Saver 类用于保存模型.
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	# 将模型保存到 /path/to/model/model.ckpt 文件
	##saver.save(sess, "/path/to/model/model.ckpt")
	saver.save(sess, "chap-5_model.ckpt")#chap-5.4.1_model.ckpt")

'''





#-----------------------------------------------
# 以下代码中给出加载这个已经保存的TensorFlow模型的方法。
# 两段代码唯一不同的是，在加载模型的代码中没有运行变量的初始化过程，而是将变量的值通过已经保存的模型加载进来。
#-----------------------------------------------
'''
import tensorflow as tf

# 使用和保存模型代码中一样的方式来声明变量。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
	# 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法。
	##saver.restore(sess, "/path/to/model/model.ckpt")
	saver.restore(sess, "chap-5.4.1_model.ckpt")
	print sess.run(result)

'''




#-----------------------------------------------
# 如果不希望重复定义图上的计算，也可以直接加载已经持久化的图。以下代码给出了一个样例。
#-----------------------------------------------
'''
import tensorflow as tf
# 直接加载持久化的图。
##saver = tf.train.import_meta_graph(
##			"/path/to/model/model.ckpt/model.ckpt.meta")
saver = tf.train.import_meta_graph("chap-5_model.ckpt.meta")#chap-5_model.ckpt/chap-5_model.ckpt.meta")
with tf.Session() as sess:
	##saver.restore(sess, "/path/to/model/model.ckpt")
	saver.restore(sess, "chap-5_model.ckpt")
	# 通过张量的名称来获取张量。
	print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
	# 输出 [3.]

'''




#--------------------------------------------------
# 上例默认保存和加载了TensorFlow计算图上定义的全部变量。但有时可能只需要保存或者加载部分变量。比如可能有一个之前训练好的五层NN模型，现在想尝试一个六层的NN，那么可以将前面五层NN中的参数直接加载到新的模型，而仅仅将最后一层NN重新训练。
# 为了保存或者加载部分变量，在声明 tf.train.Saver 类时可以提供一个列表来指定需要保存或者加载的变量。比如在加载模型的代码中使用 saver = tf.train.Saver([v1]) 命令来构建 tf.train.Saver 类，那么只有变量 v1 会被加载进来。如果运行修改后只加载了 v1 的代码会得到变量未初始化的错误：
# tensorflow.python.framework.errors.FailedPreconditionError: Attempting to use uninitialized value v2
# 因为 v2 没有被加载，所以 v2 在运行初始化之前是没有值的。除了可以选取需要被加载的变量，tf.train.Saver 类也支持在保存或者加载时给变量重命名。下面给出了一个简单的样例程序说明变量重命名是如何被使用的。
#--------------------------------------------------
'''
import tensorflow as tf

# 这里声明的变量名称和已经保存的模型中变量的名称不同。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[2]), name="other-v2")

# 如果直接使用 tf.train.Saver() 来加载模型会报变量找不到的错误。下面显示了报错信息：
# tensorflow.python.framework.errors.NotFoundError: Tensor name "other-v2" not found in checkpoint files /path/to/model/model.ckpt

# 使用一个字典（dictionary）来重命名变量就可以加载原来的模型了。这个字典指定了原来名称为 v1 的变量现在加载到变量 v1 中（名称为 other-v1），名称为 v2 的变量加载到变量 v2 中（名称为 other-v2）。
saver = tf.train.Saver({"v1":v1, "v2":v2})

'''




#--------------------------------------------
# 在这个程序中，对变量 v1 和 v2 的名称进行了修改。为解决保存时变量名称与加载时变量名称不一致的问题，TensorFlow 可以通过字典将模型保存时的变量名和需要加载的变量联系起来。
# 这样做主要目的之一是方便使用变量的滑动平均值。在4.4.3小节中介绍了使用变量的滑动平均值可以让NN模型更加健壮 (robust)。在TensorFlow中，每一个变量的滑动平均值是通过影子变量维护的，所以要获取变量的滑动平均值实际上就是获取这个影子变量的取值。如果在加载模型时直接将影子变量映射到变量自身，那么在使用训练好的模型时就不需要再调用函数来获取变量的滑动平均值了。这样大大方便了滑动平均模型的使用。以下代码给出了一个保存滑动平均模型的样例。
#--------------------------------------------
'''
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有申明滑动平均模型时只有一个变量 v，所以下面的语句只会输出 "v:0"。
for variables in tf.all_variables():
	print variables.name

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.all_variables())
# 在申明滑动平均模型之后，TensorFlow 会自动生成一个影子变量 v/ExponentialMoving Average。于是下面的语句会输出 "v:0" 和 "v/ExponentialMovingAverage:0"。
for variables in tf.all_variables():
	print variables.name

saver = tf.train.Saver()
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	sess.run(tf.assign(v, 10))
	sess.run(maintain_averages_op)
	# 保存时，TensorFlow 会将 v:0 和 v/ExponentialMovingAverage:0 两个变量都存下来。
	##saver.save(sess, "/path/to/model/model.ckpt")
	saver.save(sess, "chap-5_model.ckpt")
	print sess.run([v, ema.average(v)])
	# 输出 [10.0, 0.099999905]

'''




#--------------------------------------------
# 以下代码给出了如何通过变量重命名直接读取变量的滑动平均值。从下面程序的输出可以看出，读取的变量 v 的值实际上是上面代码中变量 v 的滑动平均值。通过这个方法，就可以使用完全一样的代码来计算滑动平均模型前向传播的结果。
#--------------------------------------------
'''
v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量 v 的滑动平均值直接赋值给 v。
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
	saver.restore(sess, "/path/to/model/model/ckpt")
	print sess.run(v)
	# 输出 0.099999905，这个值就是原来模型中变量 v 的滑动平均值。

'''




#--------------------------------------------
# 为了方便加载时重命名滑动平均变量，tf.train.ExponentialMovingAverage 类提供了 variables_to_restore 函数来生成 tf.trian.Saver 类所需要的变量重命名字典。以下代码给出了 variables_to_restore 函数的使用样例。
#--------------------------------------------
'''
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用 variables_to_restore 函数可以直接生成上面代码中提供的字典 {"v/ExponentialMovingAverage": v}。
# 以下代码会输出：
# {'v/ExponentialMovingAverage': <tensorflow.python.ops.variables.Variable object at 0x7ff6454ddc10>}
# 其中后面的 Variable 类就代表了变量 v.
print ema.variables_to_restore()

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
	##saver.restore(sess, "/path/to/model/model.ckpt")
	saver.restore(sess, "chap-5_model.ckpt")
	print sess.run(v)  ##0.0999999
	# 输出 0.099999905，即原来模型中变量 v 的滑动平均值。

'''




#-------------------------------------------
#-------------------------------------------










#-------------------------------------------
#-------------------------------------------






