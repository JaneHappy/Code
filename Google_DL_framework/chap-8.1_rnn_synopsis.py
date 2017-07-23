# coding: utf-8
#  chapter 8:  RNN (intro)




import numpy as np 

X = [1, 2]  #my: [1.,2.]
state = [0.0, 0.0]

# 分开定义不同的输入部分的权重以方便操作。
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数。
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程。
for i in range(len(X)):

	# 计算循环体中的全连接层神经网络。
	before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
	state = np.tanh(before_activation)

	# 根据当前时刻状态计算最终输出。
	final_output = np.dot(state, w_output) + b_output

	# 输出每个时刻的信息。
	print "before activation: ", before_activation
	print "state  :           ", state
	print "output :           ", final_output

'''
运行以上程序可以得到输出：
该输出和 图8-5 中的数字是一致的。

before activation:  [ 0.6  0.5]
state  :            [ 0.53704957  0.46211716]
output :            [ 1.56128388]
before activation:  [ 1.2923401   1.39225678]
state  :            [ 0.85973818  0.88366641]
output :            [ 2.72707101]
[Finished in 0.2s]
'''




'''
numpy.dot(a, b, out=None)
Dot product of two arrays.
|->   that is matrix multiply
'''



