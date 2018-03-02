# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/06_Neural_Networks/02_Implementing_an_Operational_Gate/02_gates.ipynb

from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from tensorflow.python.framework import ops 
ops.reset_default_graph()



## Gate 1
#  f(x)=a*x
#	 Start Graph Session
sess = tf.Session()

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

multiplication = tf.multiply(a, x_data)

#	 Declare the loss function as the difference between
#	 the output and a target value, 50.
loss = tf.square(tf.subtract(multiplication, 50.))

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

#	 Run loop across gate
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
	sess.run(train_step, feed_dict={x_data: x_val})
	a_val = sess.run(a)
	mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
	print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
	print('\t Loss = ', sess.run(loss, feed_dict={x_data: x_val}))



'''
2018-03-02 10:14:55.549393: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Optimizing a Multiplication Gate Output to 50.
str(a_val * 5.0 = 35.0
str(a_val * 5.0 = 42.5
str(a_val * 5.0 = 46.25
str(a_val * 5.0 = 48.125
str(a_val * 5.0 = 49.0625
str(a_val * 5.0 = 49.5312
str(a_val * 5.0 = 49.7656
str(a_val * 5.0 = 49.8828
str(a_val * 5.0 = 49.9414
str(a_val * 5.0 = 49.9707
[Finished in 11.7s]
'''



## Gate 2
#  f(x) = a*x+b
#	 Start a New Graph Session
ops.reset_default_graph()
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.multiply(a, x_data), b)

#	 Declare the loss function as the difference between
#	 the output and a target value, 50.
loss = tf.square(tf.subtract(two_gate, 50.))

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

#	 Run loop across gate
print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
	sess.run(train_step, feed_dict={x_data: x_val})
	a_val, b_val = (sess.run(a), sess.run(b))
	two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
	print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))
	print('\t Loss = ', sess.run(loss, feed_dict={x_data: x_val}))



'''
2018-03-02 10:17:21.621052: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Optimizing a Multiplication Gate Output to 50.
7.0 * 5.0 = 35.0
8.5 * 5.0 = 42.5
9.25 * 5.0 = 46.25
9.625 * 5.0 = 48.125
9.8125 * 5.0 = 49.0625
9.90625 * 5.0 = 49.5312
9.95312 * 5.0 = 49.7656
9.97656 * 5.0 = 49.8828
9.98828 * 5.0 = 49.9414
9.99414 * 5.0 = 49.9707

Optimizing Two Gate Output to 50.
5.4 * 5.0 + 1.88 = 28.88
7.512 * 5.0 + 2.3024 = 39.8624
8.52576 * 5.0 + 2.50515 = 45.134
9.01236 * 5.0 + 2.60247 = 47.6643
9.24593 * 5.0 + 2.64919 = 48.8789
9.35805 * 5.0 + 2.67161 = 49.4619
9.41186 * 5.0 + 2.68237 = 49.7417
9.43769 * 5.0 + 2.68754 = 49.876
9.45009 * 5.0 + 2.69002 = 49.9405
9.45605 * 5.0 + 2.69121 = 49.9714
[Finished in 9.0s]



2018-03-02 10:18:55.801020: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Optimizing a Multiplication Gate Output to 50.
7.0     * 5.0 = 35.0	 Loss =  225.0
8.5     * 5.0 = 42.5	 Loss =   56.25
9.25    * 5.0 = 46.25	 Loss =   14.0625
9.625   * 5.0 = 48.125	 Loss =    3.51562
9.8125  * 5.0 = 49.0625	 Loss =    0.878906
9.90625 * 5.0 = 49.5312	 Loss =    0.219727
9.95312 * 5.0 = 49.7656	 Loss =    0.0549316
9.97656 * 5.0 = 49.8828	 Loss =    0.0137329
9.98828 * 5.0 = 49.9414	 Loss =    0.00343323
9.99414 * 5.0 = 49.9707	 Loss =    0.000858307

Optimizing Two Gate Output to 50.
5.4     * 5.0 + 1.88    = 28.88 	 Loss =  446.054
7.512   * 5.0 + 2.3024  = 39.8624	 Loss =  102.771
8.52576 * 5.0 + 2.50515 = 45.134	 Loss =   23.6784
9.01236 * 5.0 + 2.60247 = 47.6643	 Loss =    5.45552
9.24593 * 5.0 + 2.64919 = 48.8789	 Loss =    1.25695
9.35805 * 5.0 + 2.67161 = 49.4619	 Loss =    0.289602
9.41186 * 5.0 + 2.68237 = 49.7417	 Loss =    0.0667232
9.43769 * 5.0 + 2.68754 = 49.876	 Loss =    0.0153733
9.45009 * 5.0 + 2.69002 = 49.9405	 Loss =    0.00354226
9.45605 * 5.0 + 2.69121 = 49.9714	 Loss =    0.000815928
[Finished in 9.4s]
'''




