# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/05_Implementing_Deming_Regression

from __future__ import division
from __future__ import print_function




# Implementing Deming Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()



#	 Load the data
#	 iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris   = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data]) # Petal Width
y_vals = np.array([y[0] for y in iris.data]) # Sepal Length

#	 Declare batch size
batch_size = 125

#	 Initialize placeholders
x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#	 Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

#	 Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b) 	#[None,1]

#	 Declare Demming loss function
demming_numerator   = tf.abs(tf.subtract(tf.add(tf.matmul(x_data, A), b), y_target))  # a number (x)
#					tf.matmul(x_data, A) 		[None,1]
#					tf.add(., b) 				[None,1]
#					tf.subtract(., y_target)	[None,1]
#					tf.abs(.) 					[None,1]
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1)) 	#[1,1]
loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

#	 Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.25)
train_step = my_opt.minimize(loss)

#	 Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#	 Training loop
loss_vec = []
for i in range(1500):
	index  = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.transpose([ x_vals[index] ])
	rand_y = np.transpose([ y_vals[index] ])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	if (i+1)%100==0:
		print('Step #'+str(i+1) + ' \t A = '+str(sess.run(A)) + ' \t b = '+str(sess.run(b)) + ' \t Loss = '+str(temp_loss))


'''
2018-03-01 17:51:03.652491: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #100 	 A = [[ 2.70096231]] 	 b = [[ 2.28448606]] 	 Loss = 0.39172
Step #200 	 A = [[ 2.05066156]] 	 b = [[ 3.20546079]] 	 Loss = 0.379725
Step #300 	 A = [[ 1.12784493]] 	 b = [[ 4.47407627]] 	 Loss = 0.27572
Step #400 	 A = [[ 0.98973441]] 	 b = [[ 4.59587288]] 	 Loss = 0.277591
Step #500 	 A = [[ 1.01736152]] 	 b = [[ 4.5854888]] 	 Loss = 0.261085
Step #600 	 A = [[ 1.0361383]] 	 b = [[ 4.6161623]] 	 Loss = 0.262333
Step #700 	 A = [[ 1.01169169]] 	 b = [[ 4.57974434]] 	 Loss = 0.279065
Step #800 	 A = [[ 1.03220308]] 	 b = [[ 4.62413549]] 	 Loss = 0.256883
Step #900 	 A = [[ 1.00623262]] 	 b = [[ 4.65293407]] 	 Loss = 0.252833
Step #1000 	 A = [[ 0.99745113]] 	 b = [[ 4.58868933]] 	 Loss = 0.259299
Step #1100 	 A = [[ 1.01672995]] 	 b = [[ 4.62545729]] 	 Loss = 0.307717
Step #1200 	 A = [[ 1.03476012]] 	 b = [[ 4.64517879]] 	 Loss = 0.276264
Step #1300 	 A = [[ 1.03128386]] 	 b = [[ 4.61854982]] 	 Loss = 0.259712
Step #1400 	 A = [[ 0.99907523]] 	 b = [[ 4.58338833]] 	 Loss = 0.273587
Step #1500 	 A = [[ 1.01084554]] 	 b = [[ 4.63948202]] 	 Loss = 0.30718
[Finished in 15.9s]
'''


#	 Get the optimal coefficients
#[slope] = sess.run(A)
#[y_intercept] = sess.run(b)
[[slope]]       = sess.run(A)
[[y_intercept]] = sess.run(b)

#	 Get best fit line
best_fit = []
for i in x_vals:
	best_fit.append(slope*i + y_intercept)

plt.figure(figsize=(12, 5))
plt.subplot(121)
#	 Plot the result
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width' )
plt.ylabel('Sepal Length')
#plt.show()

plt.subplot(122)
#	 Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Demming Loss per Generation')
plt.xlabel('Iteration')
plt.ylabel('Demming Loss')
#plt.show()


'''
2018-03-01 17:57:21.703441: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #100 	 A = [[-5.42457485]] 	 b = [[  5.66404295]] 	 Loss = 1.22173
Step #200 	 A = [[-5.25228596]] 	 b = [[  7.93181896]] 	 Loss = 0.88552
Step #300 	 A = [[-4.68121529]] 	 b = [[  9.55619144]] 	 Loss = 0.86286
Step #400 	 A = [[-4.01213503]] 	 b = [[ 10.78954887]] 	 Loss = 0.836494
Step #500 	 A = [[-4.22311211]] 	 b = [[ 11.16029549]] 	 Loss = 0.792966
Step #600 	 A = [[-4.4047246]] 	 b = [[ 11.50782871]] 	 Loss = 0.75336
Step #700 	 A = [[-4.68690109]] 	 b = [[ 11.73652363]] 	 Loss = 0.800559
Step #800 	 A = [[-4.87125444]] 	 b = [[ 12.00315285]] 	 Loss = 0.840549
Step #900 	 A = [[-5.04871988]] 	 b = [[ 12.24726009]] 	 Loss = 0.818471
Step #1000 	 A = [[-5.22532177]] 	 b = [[ 12.476964  ]] 	 Loss = 0.738957
Step #1100 	 A = [[-5.37845087]] 	 b = [[ 12.70724487]] 	 Loss = 0.800397
Step #1200 	 A = [[-5.55522108]] 	 b = [[ 12.90196514]] 	 Loss = 0.756181
Step #1300 	 A = [[-5.69403505]] 	 b = [[ 13.10246277]] 	 Loss = 0.759506
Step #1400 	 A = [[-5.8333106]] 	 b = [[ 13.29696274]] 	 Loss = 0.704555
Step #1500 	 A = [[-5.98176908]] 	 b = [[ 13.46836185]] 	 Loss = 0.778237
[Finished in 33.1s]
'''




#-------------------
# My
#-------------------

ops.reset_default_graph()
sess = tf.Session()
learning_rate = 0.25
batch_size    = 125 #25

num_samples = len(iris.data)
num_feat    = len(iris.data[0]) - 1
x_vals = iris.data[:, 1:]  #[num_samples, 3], 3=num_feature
y_vals = iris.data[:, :1]  #[num_samples, 1]

x_data   = tf.placeholder(shape=[None, num_feat], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1	   ], dtype=tf.float32)

ones_col = np.array([ np.repeat(1., num_samples) ]).T 
Ax_vals  = np.concatenate((x_vals, ones_col), axis=1)  #[num_samples, num_feat+1]
'''
[Ax]c = y
'''

Ax_data  = tf.placeholder(shape=[None, num_feat+1], dtype=tf.float32)

# Linear regression
lin_reg_coef = tf.Variable(tf.random_normal(shape=[num_feat+1, 1]))
lin_reg_model_output = tf.matmul(Ax_data, lin_reg_coef)
lin_reg_loss         = tf.reduce_mean(tf.square(y_target - lin_reg_model_output))
lin_reg_my_opt       = tf.train.GradientDescentOptimizer(learning_rate)
lin_reg_train_step   = lin_reg_my_opt.minimize(lin_reg_loss)

demming_coef = tf.Variable(tf.random_normal(shape=[num_feat+1, 1]))  # include a bias
'''
demming_numerator    = tf.abs(tf.matmul(Ax_data, demming_coef) - y_target)
#			tf.matmul(.,.)		[None, 1]
#			tf.subtract(.,.)	[None, 1]
#			tf.abs(.,.)			[None, 1]
demming_denominator  = tf.sqrt(tf.reduce_sum(tf.square(demming_coef)))
#			tf.square(.) 		[num_feat+1, 1]
#			tf.reduce_sum(.)	a number
#			tf.sqrt(.)			a number
demming_model_output = tf.truediv(demming_numerator, demming_denominator)  #[None, 1]
demming_loss         = tf.reduce_mean(tf.square(y_target - demming_model_output))
'''
demming_model_output = tf.matmul(Ax_data, demming_coef)
demming_numerator    = tf.abs(tf.subtract(demming_model_output, y_target))
demming_denominator  = tf.sqrt(tf.reduce_sum(tf.square(demming_coef)))
demming_loss         = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))
demming_my_opt       = tf.train.GradientDescentOptimizer(learning_rate)
demming_train_step   = demming_my_opt.minimize(demming_loss)

# init
sess.run(tf.global_variables_initializer())

# training loop
lin_reg_loss_vec = []
demming_loss_vec = []
for i in range(1500):
	index  = np.random.choice(num_samples, size=batch_size)
	rand_x = Ax_vals[index]  #[batch_size, num_feat+1]
	rand_y = y_vals[ index]  #[batch_size, 1 		 ]

	#lin_reg
	sess.run(lin_reg_train_step, feed_dict={Ax_data: rand_x, y_target: rand_y})
	temp_lin_reg_loss = sess.run(lin_reg_loss, feed_dict={Ax_data: rand_x, y_target: rand_y})
	lin_reg_loss_vec.append(temp_lin_reg_loss)

	#demming
	sess.run(demming_train_step, feed_dict={Ax_data: rand_x, y_target: rand_y})
	temp_demming_loss = sess.run(demming_loss, feed_dict={Ax_data: rand_x, y_target: rand_y})
	demming_loss_vec.append(temp_demming_loss)

	if (i+1)%100==0:
		print('Step #' + str(i+1))
		#print('\t lin_reg: ' + '\t coef = '+str(sess.run(lin_reg_coef)) + '\t Loss = '+str(temp_lin_reg_loss))
		#print('\t demming: ' + '\t coef = '+str(sess.run(demming_coef)) + '\t Loss = '+str(temp_demming_loss))
		print('\t lin_reg: ' + '\t coef = '+str(sess.run(tf.squeeze(lin_reg_coef))) + '\t Loss = '+str(temp_lin_reg_loss))
		print('\t demming: ' + '\t coef = '+str(sess.run(tf.squeeze(demming_coef))) + '\t Loss = '+str(temp_demming_loss))


lin_reg_eval = sess.run(tf.squeeze(lin_reg_model_output), feed_dict={Ax_data: Ax_vals, y_target: y_vals})
demming_eval = sess.run(tf.squeeze(demming_model_output), feed_dict={Ax_data: Ax_vals, y_target: y_vals})
original_label = y_vals.T[0]  #or: y_vals[:,0]

plt.figure(figsize=(12, 5))  #7, 6.5))
plt.subplot(121)
plt.plot(x_vals[:,0], original_label, 'ro', label='Data Points')
plt.plot(x_vals[:,0], lin_reg_eval, 'b*', label='lin_reg eval')
plt.plot(x_vals[:,0], demming_eval, 'g*', label='demming eval')
plt.xlabel('x_vals[:,0]')
plt.ylabel('y_vals')
plt.legend()
plt.title('Data Points')

plt.subplot(122)
plt.plot(lin_reg_loss_vec, 'b-', label='lin_reg loss')
plt.plot(demming_loss_vec, 'g-', label='demming loss')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
#plt.show()



'''
exe2: batch_size=25
2018-03-01 18:53:33.482784: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #100 	 A = [[-6.44518566]] 	 b = [[ 3.91501832]] 	 Loss = 1.45583
Step #200 	 A = [[-6.62723398]] 	 b = [[ 7.01275635]] 	 Loss = 1.07144
Step #300 	 A = [[-6.36044359]] 	 b = [[ 8.59703064]] 	 Loss = 0.976439
Step #400 	 A = [[-5.86300564]] 	 b = [[ 9.96472931]] 	 Loss = 0.944075
Step #500 	 A = [[-5.09914112]] 	 b = [[ 11.26289272]] 	 Loss = 0.799814
Step #600 	 A = [[-4.79735756]] 	 b = [[ 11.94095516]] 	 Loss = 0.704356
Step #700 	 A = [[-5.02034092]] 	 b = [[ 12.16566467]] 	 Loss = 0.664068
Step #800 	 A = [[-5.16698027]] 	 b = [[ 12.42504787]] 	 Loss = 0.768001
Step #900 	 A = [[-5.34045744]] 	 b = [[ 12.63776398]] 	 Loss = 0.75943
Step #1000 	 A = [[-5.48983622]] 	 b = [[ 12.85729218]] 	 Loss = 0.794822
Step #1100 	 A = [[-5.69080877]] 	 b = [[ 13.0148592]] 	 Loss = 0.807739
Step #1200 	 A = [[-5.79827976]] 	 b = [[ 13.23309803]] 	 Loss = 0.781684
Step #1300 	 A = [[-5.92796612]] 	 b = [[ 13.42206383]] 	 Loss = 0.784711
Step #1400 	 A = [[-6.0805378]] 	 b = [[ 13.58061886]] 	 Loss = 0.769293
Step #1500 	 A = [[-6.20614338]] 	 b = [[ 13.75152779]] 	 Loss = 0.689212
Step #100
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.60039496] [-1.05787027] [ 0.34346521] [-1.98332286]]	 Loss = 0.0517077
Step #200
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.21500766] [-0.96386278] [ 0.53103983] [-1.86242902]]	 Loss = 0.0530101
Step #300
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.15279317] [-0.94126201] [ 0.60025954] [-1.80217576]]	 Loss = 0.0176684
Step #400
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.25729084] [-1.05402935] [ 0.63095713] [-1.82692468]]	 Loss = 0.0307884
Step #500
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.20154798] [-0.95290697] [ 0.74046361] [-1.82825756]]	 Loss = 0.0570018
Step #600
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.25220752] [-1.07292247] [ 0.63981646] [-1.68828571]]	 Loss = 0.0833837
Step #700
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
 	 demming: 	 coef = [[-1.12712669] [-0.94354248] [ 0.76413286] [-1.73059821]]	 Loss = 0.0427075
Step #800
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.21657884] [-1.00711584] [ 0.76883698] [-1.72015023]]	 Loss = 0.0246342
Step #900
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.25477922] [-1.10192335] [ 0.69670808] [-1.66036904]]	 Loss = 0.0401273
Step #1000
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.20826983] [-1.09041893] [ 0.64296854] [-1.59192717]]	 Loss = 0.0998253
Step #1100
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.25914454] [-1.05972278] [ 0.73243332] [-1.70228624]]	 Loss = 0.0250005
Step #1200
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.23381269] [-1.0297159 ] [ 0.79505879] [-1.73724508]]	 Loss = 0.0456984
Step #1300
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.17191708] [-1.03532183] [ 0.7546435 ] [-1.70094132]]	 Loss = 0.0239852
Step #1400
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.26873684] [-1.05542815] [ 0.70002109] [-1.67678261]]	 Loss = 0.0312412
Step #1500
	 lin_reg: 	 coef = [[ nan] [ nan] [ nan] [ nan]]	 Loss = nan
	 demming: 	 coef = [[-1.33799553] [-1.13136446] [ 0.74313462] [-1.72078383]]	 Loss = 0.0445777
[Finished in 68.8s]

exe3: batch_size=125
2018-03-01 18:59:02.327565: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #100 	 A = [[-4.42278671]] 	 b = [[ 6.64704895]] 	 Loss = 1.13865
Step #200 	 A = [[-3.96112633]] 	 b = [[ 8.70043564]] 	 Loss = 0.875752
Step #300 	 A = [[-3.40157008]] 	 b = [[ 10.04637146]] 	 Loss = 0.83231
Step #400 	 A = [[-3.72384787]] 	 b = [[ 10.46005821]] 	 Loss = 0.789605
Step #500 	 A = [[-3.98485732]] 	 b = [[ 10.8441143]] 	 Loss = 0.78002
Step #600 	 A = [[-4.22078276]] 	 b = [[ 11.19429398]] 	 Loss = 0.667158
Step #700 	 A = [[-4.449049]] 	 b = [[ 11.49887085]] 	 Loss = 0.735802
Step #800 	 A = [[-4.68363476]] 	 b = [[ 11.75440121]] 	 Loss = 0.741485
Step #900 	 A = [[-4.88051987]] 	 b = [[ 12.01657295]] 	 Loss = 0.740433
Step #1000 	 A = [[-5.03951359]] 	 b = [[ 12.27538967]] 	 Loss = 0.793343
Step #1100 	 A = [[-5.24980545]] 	 b = [[ 12.47373676]] 	 Loss = 0.813967
Step #1200 	 A = [[-5.37749195]] 	 b = [[ 12.71785545]] 	 Loss = 0.720088
Step #1300 	 A = [[-5.57873869]] 	 b = [[ 12.88623142]] 	 Loss = 0.794397
Step #1400 	 A = [[-5.70007992]] 	 b = [[ 13.10560131]] 	 Loss = 0.726974
Step #1500 	 A = [[-5.83884716]] 	 b = [[ 13.29251957]] 	 Loss = 0.747562
Step #100
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-3.96305537 -2.01454139 -0.0605     -1.22810352]	 Loss = 0.102898
Step #200
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-3.59373713 -1.86045301  0.24329284 -1.30018091]	 Loss = 0.102164
Step #300
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-3.2403059  -1.84927523  0.509561   -1.38824964]	 Loss = 0.0661541
Step #400
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-2.8627243  -1.76642919  0.72678876 -1.44644463]	 Loss = 0.0920697
Step #500
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-2.45570922 -1.64469802  0.90092891 -1.47092092]	 Loss = 0.0621837
Step #600
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-2.04256082 -1.48466659  0.97717226 -1.44789064]	 Loss = 0.0592998
Step #700
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.63246262 -1.32533979  1.02779698 -1.41720712]	 Loss = 0.0319884
Step #800
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.23729551 -1.13785565  1.06993198 -1.39984024]	 Loss = 0.02983
Step #900
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.135198   -1.08624303  0.96322215 -1.39540565]	 Loss = 0.0382432
Step #1000
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.23066795 -1.12785196  0.92817229 -1.48744476]	 Loss = 0.034036
Step #1100
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.15850353 -1.03094697  0.96760148 -1.5429548 ]	 Loss = 0.0586624
Step #1200
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.1774292  -1.10332835  0.83519983 -1.52148509]	 Loss = 0.049803
Step #1300
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.13493955 -1.05436027  0.82827038 -1.53889537]	 Loss = 0.0439997
Step #1400
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.12153912 -1.05842423  0.76215875 -1.52356803]	 Loss = 0.072226
Step #1500
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-1.17355692 -1.07830346  0.75877047 -1.56926942]	 Loss = 0.0640251
[Finished in 40.6s]
'''




#----------------------
# My
#----------------------

#plt.close('all')
ops.reset_default_graph()
sess = tf.Session()
batch_size = 125

x_vals = iris.data[:,3]  # Petal Width
y_vals = iris.data[:,0]  # Sepal Length

x_data   = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#lin_reg
lin_reg_A = tf.Variable(tf.random_normal(shape=[1, 1]))
lin_reg_b = tf.Variable(tf.random_normal(shape=[1, 1]))
lin_reg_model_output = tf.add(tf.matmul(x_data, lin_reg_A), lin_reg_b) 				#[None,1]
lin_reg_loss         = tf.reduce_mean(tf.square(y_target - lin_reg_model_output)) 	#[None,1] -> a number
lin_reg_my_opt       = tf.train.GradientDescentOptimizer(learning_rate)
lin_reg_train_step   = lin_reg_my_opt.minimize(lin_reg_loss)

#demming
demming_A = tf.Variable(tf.random_normal(shape=[1, 1]))
demming_b = tf.Variable(tf.random_normal(shape=[1, 1]))
#demming_numerator   = tf.abs(tf.subtract(tf.add(tf.matmul(x_data, demming_A), demming_b), y_target))  #[None,1]
#demming_denominator = tf.sqrt(tf.add(tf.square(tf.squeeze(A)), 1.))  #a number #error!
demming_model_output = tf.add(tf.matmul(x_data, demming_A), demming_b)  	#[None,1]
demming_numerator    = tf.abs(tf.subtract(demming_model_output, y_target))  #[None,1]
demming_denominator  = tf.sqrt(tf.add(tf.square(demming_A), 1)) 	#[1,1]
demming_loss         = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))  #[None,1] -> a number
demming_my_opt       = tf.train.GradientDescentOptimizer(learning_rate)
demming_train_step   = demming_my_opt.minimize(demming_loss)

sess.run(tf.global_variables_initializer())

lin_reg_loss_vec = []
demming_loss_vec = []
for i in range(1500):
	index  = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.transpose([ x_vals[index] ])
	rand_y = np.transpose([ y_vals[index] ])

	#lin_reg
	sess.run(lin_reg_train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss1 = sess.run(lin_reg_loss, feed_dict={x_data: rand_x, y_target: rand_y})
	lin_reg_loss_vec.append(temp_loss1)

	#demming
	sess.run(demming_train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss2 = sess.run(demming_loss, feed_dict={x_data: rand_x, y_target: rand_y})
	demming_loss_vec.append(temp_loss2)

	if (i+1)%100==0:
		print('Step #'+str(i+1))
		print('\t lin_reg: ' + '\t A = '+str(sess.run(lin_reg_A)) + '\t b = '+str(sess.run(lin_reg_b)) + '\t Loss = '+str(temp_loss1))
		print('\t demming: ' + '\t A = '+str(sess.run(demming_A)) + '\t b = '+str(sess.run(demming_b)) + '\t Loss = '+str(temp_loss1))


eval_x = np.array([ x_vals ]).T 
eval_y = np.array([ y_vals ]).T 
lin_reg_eval = sess.run(tf.squeeze(lin_reg_model_output), feed_dict={x_data: eval_x, y_target: eval_y})
demming_eval = sess.run(tf.squeeze(demming_model_output), feed_dict={x_data: eval_x, y_target: eval_y})

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(x_vals, y_vals, 'ro', label='Data Points')
plt.plot(x_vals, lin_reg_eval, 'g*', label='lin_reg eval')
plt.plot(x_vals, demming_eval, 'b*', label='demming eval')
plt.xlabel('x_vals')
plt.ylabel('y_vals')
plt.legend(loc='upper left')

plt.subplot(122)
plt.plot(lin_reg_loss_vec, 'g-', label='lin_reg loss')
plt.plot(demming_loss_vec, 'b-', label='demming loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


'''
exe4:

2018-03-01 19:42:15.907602: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
Step #100 	 A = [[-5.09591627]] 	 b = [[ 6.0487957]] 	 Loss = 1.28829
Step #200 	 A = [[-4.87647581]] 	 b = [[ 8.12417984]] 	 Loss = 0.995543
Step #300 	 A = [[-4.11071777]] 	 b = [[ 9.85012436]] 	 Loss = 0.792525
Step #400 	 A = [[-3.84025311]] 	 b = [[ 10.70511627]] 	 Loss = 0.800446
Step #500 	 A = [[-4.14013529]] 	 b = [[ 11.02175045]] 	 Loss = 0.808754
Step #600 	 A = [[-4.38207293]] 	 b = [[ 11.33889103]] 	 Loss = 0.784886
Step #700 	 A = [[-4.59879017]] 	 b = [[ 11.63080311]] 	 Loss = 0.856063
Step #800 	 A = [[-4.80448723]] 	 b = [[ 11.89350414]] 	 Loss = 0.806754
Step #900 	 A = [[-4.98380899]] 	 b = [[ 12.14936352]] 	 Loss = 0.755252
Step #1000 	 A = [[-5.14696026]] 	 b = [[ 12.39457703]] 	 Loss = 0.755821
Step #1100 	 A = [[-5.33353615]] 	 b = [[ 12.60334873]] 	 Loss = 0.735396
Step #1200 	 A = [[-5.47027254]] 	 b = [[ 12.83498764]] 	 Loss = 0.766106
Step #1300 	 A = [[-5.6370883]] 	 b = [[ 13.02161217]] 	 Loss = 0.693739
Step #1400 	 A = [[-5.78278494]] 	 b = [[ 13.21671486]] 	 Loss = 0.78285
Step #1500 	 A = [[-5.90344191]] 	 b = [[ 13.41197014]] 	 Loss = 0.791455

Step #100
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [ 0.47523794  0.07611346  1.10885358  3.68402719]	 Loss = 0.242359
Step #200
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [ 0.1793755   0.01499444  1.161273    4.38188457]	 Loss = 0.135854
Step #300
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [ 0.03811188 -0.01715454  1.01401222  4.75494671]	 Loss = 0.100198
Step #400
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.0560641   0.05879442  0.91474026  5.06326246]	 Loss = 0.0948476
Step #500
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.14841495  0.01907459  0.76307493  5.24910307]	 Loss = 0.0719134
Step #600
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.15739968  0.08050738  0.69034249  5.39896584]	 Loss = 0.0731325
Step #700
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.19498925  0.11140592  0.64351046  5.5214839 ]	 Loss = 0.0844435
Step #800
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.27452534  0.07352541  0.57932228  5.63092566]	 Loss = 0.0737025
Step #900
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.3044816   0.11402754  0.56804079  5.73307133]	 Loss = 0.065637
Step #1000
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.35422537  0.05999451  0.54059118  5.79763365]	 Loss = 0.0761433
Step #1100
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.33869427  0.0919001   0.55428672  5.8842082 ]	 Loss = 0.0796076
Step #1200
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.38105398  0.01034793  0.53406852  5.94065189]	 Loss = 0.08092
Step #1300
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.39231402  0.06451543  0.55175483  5.99401712]	 Loss = 0.0643146
Step #1400
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.40797246  0.04390627  0.54069138  6.0520215 ]	 Loss = 0.070563
Step #1500
	 lin_reg: 	 coef = [ nan  nan  nan  nan]	 Loss = nan
	 demming: 	 coef = [-0.43029818  0.03710804  0.55035371  6.08935595]	 Loss = 0.0680463

Step #100
	 lin_reg: 	 A = [[ 0.96644008]]	 b = [[ 4.84194565]]	 Loss = 0.25055
	 demming: 	 A = [[ 3.33881068]]	 b = [[ 1.41032135]]	 Loss = 0.25055
Step #200
	 lin_reg: 	 A = [[ 0.98415929]]	 b = [[ 4.83921528]]	 Loss = 0.226819
	 demming: 	 A = [[ 2.8278501 ]]	 b = [[ 2.13208842]]	 Loss = 0.226819
Step #300
	 lin_reg: 	 A = [[ 0.88749588]]	 b = [[ 4.74709034]]	 Loss = 0.208737
	 demming: 	 A = [[ 2.20280504]]	 b = [[ 2.96253538]]	 Loss = 0.208737
Step #400
	 lin_reg: 	 A = [[ 0.91037607]]	 b = [[ 4.81499052]]	 Loss = 0.179551
	 demming: 	 A = [[ 1.27790272]]	 b = [[ 4.23503113]]	 Loss = 0.179551
Step #500
	 lin_reg: 	 A = [[ 0.89818394]]	 b = [[ 4.79166508]]	 Loss = 0.277205
	 demming: 	 A = [[ 1.03404927]]	 b = [[ 4.60603428]]	 Loss = 0.277205
Step #600
	 lin_reg: 	 A = [[ 0.90868473]]	 b = [[ 4.82888079]]	 Loss = 0.273838
	 demming: 	 A = [[ 1.03945243]]	 b = [[ 4.64462662]]	 Loss = 0.273838
Step #700
	 lin_reg: 	 A = [[ 0.83914572]]	 b = [[ 4.77665043]]	 Loss = 0.166172
	 demming: 	 A = [[ 0.94598049]]	 b = [[ 4.58572721]]	 Loss = 0.166172
Step #800
	 lin_reg: 	 A = [[ 0.8765623 ]]	 b = [[ 4.78041506]]	 Loss = 0.212672
	 demming: 	 A = [[ 1.03415775]]	 b = [[ 4.61953068]]	 Loss = 0.212672
Step #900
	 lin_reg: 	 A = [[ 0.8548485 ]]	 b = [[ 4.7656188 ]]	 Loss = 0.220826
	 demming: 	 A = [[ 1.01348209]]	 b = [[ 4.63311243]]	 Loss = 0.220826
Step #1000
	 lin_reg: 	 A = [[ 0.93456274]]	 b = [[ 4.73777914]]	 Loss = 0.224351
	 demming: 	 A = [[ 1.04624176]]	 b = [[ 4.59507561]]	 Loss = 0.224351
Step #1100
	 lin_reg: 	 A = [[ 0.86044145]]	 b = [[ 4.75953197]]	 Loss = 0.271574
	 demming: 	 A = [[ 1.00553119]]	 b = [[ 4.58452797]]	 Loss = 0.271574
Step #1200
	 lin_reg: 	 A = [[ 0.88141537]]	 b = [[ 4.76533127]]	 Loss = 0.203108
	 demming: 	 A = [[ 0.98961294]]	 b = [[ 4.5826664 ]]	 Loss = 0.203108
Step #1300
	 lin_reg: 	 A = [[ 0.84584934]]	 b = [[ 4.75400448]]	 Loss = 0.192862
	 demming: 	 A = [[ 1.00924098]]	 b = [[ 4.58958673]]	 Loss = 0.192862
Step #1400
	 lin_reg: 	 A = [[ 0.89941043]]	 b = [[ 4.79866695]]	 Loss = 0.224968
	 demming: 	 A = [[ 1.00247264]]	 b = [[ 4.60791969]]	 Loss = 0.224968
Step #1500
	 lin_reg: 	 A = [[ 0.9618665 ]]	 b = [[ 4.8060317 ]]	 Loss = 0.222076
	 demming: 	 A = [[ 1.08557034]]	 b = [[ 4.62509346]]	 Loss = 0.222076
[Finished in 92.0s]
'''





