# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions/04_lin_reg_l1_vs_l2.ipynb

from __future__ import division
from __future__ import print_function




# Linear Regression: L1 vs L2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()

#	 iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

batch_size    = 25
learning_rate = 0.1  # Will not converge with learning rate at 0.4
iterations    = 50

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)



#-------------
# L1-Loss
#-------------

#	 Create variables for linear regression
l1_A = tf.Variable(tf.random_normal(shape=[1, 1]))
l1_b = tf.Variable(tf.random_normal(shape=[1, 1]))
#	 Declare model operations
l1_model_output = tf.add(tf.matmul(x_data, l1_A), l1_b) 		#[None,1]

#	 Declare loss functions
l1_loss = tf.reduce_mean(tf.abs(y_target - l1_model_output))	#[None,1]-> a number
#	 Declare optimizers
l1_my_opt = tf.train.GradientDescentOptimizer(learning_rate)
l1_train_step = l1_my_opt.minimize(l1_loss)


#-------------
# L2-Loss
#-------------

#	 Create variables for linear regression
l2_A = tf.Variable(tf.random_normal(shape=[1, 1]))
l2_b = tf.Variable(tf.random_normal(shape=[1, 1]))
#	 Declare model operations
l2_model_output = tf.add(tf.matmul(x_data, l2_A), l2_b)

#	 Declare loss functions
l2_loss = tf.reduce_mean(tf.square(y_target - l2_model_output))
#	 Declare optimizers
l2_my_opt = tf.train.GradientDescentOptimizer(learning_rate)
l2_train_step = l2_my_opt.minimize(l2_loss)



#-------------
# L1-Loss -- modified learning-rate
# L2-Loss -- modified learning-rate
#-------------

learning_modi  = 0.4

l1m_C = tf.Variable(tf.random_normal(shape=[1, 1]))
l1m_d = tf.Variable(tf.random_normal(shape=[1, 1]))
l1m_model_output = tf.add(tf.matmul(x_data, l1m_C), l1m_d)
l1m_loss         = tf.reduce_mean(tf.abs(y_target - l1m_model_output))
l1m_my_opt       = tf.train.GradientDescentOptimizer(learning_modi)
l1m_train_step   = l1m_my_opt.minimize(l1m_loss)

l2m_C = tf.Variable(tf.random_normal(shape=[1, 1]))
l2m_d = tf.Variable(tf.random_normal(shape=[1, 1]))
l2m_model_output = tf.add(tf.matmul(x_data, l2m_C), l2m_d)
l2m_loss         = tf.reduce_mean(tf.square(y_target - l2m_model_output))
l2m_my_opt       = tf.train.GradientDescentOptimizer(learning_modi)
l2m_train_step   = l2m_my_opt.minimize(l2m_loss)


#-------------
# L1-Loss -- same opt
# L2-Loss -- same opt
#-------------

#same_l1_A = tf.Variable(tf.random_normal(shape=[1, 1]))


#-------------
# Both
#-------------

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)


#	 Training loop
l1_loss_vec = []
l2_loss_vec = []
l1_loss_mod = []
l2_loss_mod = []

for i in range(iterations):
	idx    = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.transpose([ x_vals[idx] ])
	rand_y = np.transpose([ y_vals[idx] ])

	# L1-Loss
	sess.run(l1_train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_l1_loss = sess.run(l1_loss, feed_dict={x_data: rand_x, y_target: rand_y})
	l1_loss_vec.append(temp_l1_loss)

	# L2-loss
	sess.run(l2_train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_l2_loss = sess.run(l2_loss, feed_dict={x_data: rand_x, y_target: rand_y})
	l2_loss_vec.append(temp_l2_loss)

	# L1-Loss --
	sess.run(l1m_train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_l1m_loss = sess.run(l1m_loss, feed_dict={x_data: rand_x, y_target: rand_y}) 
	l1_loss_mod.append(temp_l1m_loss)
	# L2-Loss --
	sess.run(l2m_train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_l2m_loss = sess.run(l2m_loss, feed_dict={x_data: rand_x, y_target: rand_y})
	l2_loss_mod.append(temp_l2m_loss)

	if (i+1)%25==0:
		print('Step #' + str(i+1))
		print("\t L1:  \t" + ' A = '+str(sess.run(l1_A)) + ' b = '+str(sess.run(l1_b)) + '\t Loss = '+str(temp_l1_loss))
		print("\t L2:  \t" + ' A = '+str(sess.run(l2_A)) + ' b = '+str(sess.run(l2_b)) + '\t Loss = '+str(temp_l2_loss))
		print("\t L1m: \t" + ' C = '+str(sess.run(l1m_C)) + ' b = '+str(sess.run(l1m_d)) + '\t Loss = '+str(temp_l1m_loss))
		print("\t L2m: \t" + ' C = '+str(sess.run(l2m_C)) + ' b = '+str(sess.run(l2m_d)) + '\t Loss = '+str(temp_l2m_loss))




#-------------
# Plot
#-------------

change_x  = np.array([x_vals]).T 
change_y  = np.array([y_vals]).T
l1_y_eval = sess.run(tf.squeeze(l1_model_output), feed_dict={x_data: change_x, y_target: change_y})  #[num_samples,1] -> (num_samples)
l2_y_eval = sess.run(tf.squeeze(l2_model_output), feed_dict={x_data: change_x, y_target: change_y})
l1m_y_eval = sess.run(tf.squeeze(l1m_model_output), feed_dict={x_data: change_x, y_target: change_y})
l2m_y_eval = sess.run(tf.squeeze(l2m_model_output), feed_dict={x_data: change_x, y_target: change_y})



#	 Plot loss over time

#exe1: plt.figure(figsize=(6,6))  #(6.5, 6.5))  #(7,7)) #8,6
plt.figure(figsize=(7,6))

plt.subplot(221)
plt.plot(l1_loss_vec, 'k-',  label='L1 Loss')
plt.plot(l2_loss_vec, 'r--', label='L2 Loss')
plt.title('L1 and L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss') #'L1'
plt.legend(loc='upper right')
#plt.show()

plt.subplot(222)
plt.plot(l1_loss_mod, 'k-',  label='L1 Loss')
plt.plot(l2_loss_mod, 'r--', label='L2 Loss')
plt.title('L1 vs L2 Loss (learning rate = 0.4)')
plt.legend(loc='upper right')

plt.subplot(223)
plt.plot(x_vals, y_vals, 'ro', label='Data')
plt.plot(x_vals, l1_y_eval, 'g*', label='Eval (L1)')
plt.plot(x_vals, l2_y_eval, 'b*', label='Eval (L2)')
plt.legend()
plt.xlabel('x_vals')
plt.ylabel('y_vals')

plt.subplot(224)
plt.plot(x_vals, y_vals, 'ro', label='Data')
plt.plot(x_vals, l1m_y_eval, 'g^', label='Eval (L1+)')
plt.plot(x_vals, l2m_y_eval, 'b^', label='Eval (L2+)')
plt.legend()
plt.xlabel('x_vals')
plt.ylabel('y_vals')

plt.show()


'''
2018-03-01 16:06:01.883764: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #25
	 L1:  	 A = [[ 1.89441097]] b = [[ 2.68963265]]	 Loss = 0.980621
	 L2:  	 A = [[ 1.70866632]] b = [[ 3.57084513]]	 Loss = 0.504517
	 L1m: 	 C = [[ 1.78897119]] b = [[ 3.47161078]]	 Loss = 0.683875
	 L2m: 	 C = [[ 115.72618103]] b = [[ 102.11097717]]	 Loss = 58067.1
Step #50
	 L1:  	 A = [[ 1.60361111]] b = [[ 3.46163321]]	 Loss = 0.474618
	 L2:  	 A = [[ 1.08460045]] b = [[ 4.35835171]]	 Loss = 0.147005
	 L1m: 	 C = [[ 0.74417114]] b = [[ 4.76761007]]	 Loss = 0.294181
	 L2m: 	 C = [[-32443.83789062]] b = [[-17798.765625]]	 Loss = 5.20216e+09
[Finished in 33.9s]

2018-03-01 16:07:09.404545: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #25
	 L1:  	 A = [[ 2.36314511]] b = [[ 0.64616382]]	 Loss = 2.08388
	 L2:  	 A = [[ 1.50105453]] b = [[ 3.70962095]]	 Loss = 0.499358
	 L1m: 	 C = [[ 1.88935447]] b = [[ 2.89469862]]	 Loss = 0.862385
	 L2m: 	 C = [[ 1357.60290527]] b = [[ 948.54840088]]	 Loss = 8.3156e+06
Step #50
	 L1:  	 A = [[ 2.77074552]] b = [[ 1.84216392]]	 Loss = 1.31107
	 L2:  	 A = [[ 1.07339787]] b = [[ 4.34402323]]	 Loss = 0.317044
	 L1m: 	 C = [[ 0.69095445]] b = [[ 4.57469893]]	 Loss = 0.416489
	 L2m: 	 C = [[-365821.71875]] b = [[-200512.03125]]	 Loss = 5.03795e+11
[Finished in 20.8s]


exe1.png
2018-03-01 16:07:38.839926: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #25
	 L1:  	 A = [[   2.45292568]] 	b = [[   2.23203349]]	 Loss = 1.06083
	 L2:  	 A = [[   1.76477182]] 	b = [[   3.51813889]]	 Loss = 0.551035
	 L1m: 	 C = [[   2.10978723]] 	b = [[   3.03725123]]	 Loss = 0.770068
	 L2m: 	 C = [[ 214.15318298]] 	b = [[ 138.36056519]]	 Loss = 185205.0
Step #50
	 L1:  	 A = [[ 2.05092573]] 	b = [[ 2.93203378  ]]	 Loss = 0.988068
	 L2:  	 A = [[ 1.18876195]] 	b = [[ 4.26808119  ]]	 Loss = 0.312717
	 L1m: 	 C = [[ 0.87298739]] 	b = [[ 4.68525124  ]]	 Loss = 0.357834
	 L2m: 	 C = [[-79477.921875]] 	b = [[-38875.828125]]	 Loss = 1.82333e+10
[Finished in 31.3s]

exe2.png
2018-03-01 16:08:41.032522: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #25
	 L1:  	 A = [[   2.18236446]] 	b = [[   2.71453834]]	 Loss = 1.29973
	 L2:  	 A = [[   1.88154817]] 	b = [[   3.3623507 ]]	 Loss = 1.24285
	 L1m: 	 C = [[   1.33979511]] 	b = [[   4.16925144]]	 Loss = 0.557654
	 L2m: 	 C = [[ 639.93896484]] 	b = [[ 497.52972412]]	 Loss = 1.50514e+06
Step #50
	 L1:  	 A = [[ 1.82476437]] 	b = [[ 3.39053893  ]]	 Loss = 0.879527
	 L2:  	 A = [[ 1.18107665]] 	b = [[ 4.21542263  ]]	 Loss = 0.271226
	 L1m: 	 C = [[ 0.68379498]] 	b = [[ 4.69725132  ]]	 Loss = 0.308974
	 L2m: 	 C = [[-53067.125]] 	b = [[-37321.60546875]]	 Loss = 1.16694e+10
[Finished in 35.9s]
'''



