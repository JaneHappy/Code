# coding: utf-8
# tensorflow cookbook
#	03	Linear Regression
#		02	Implementing a Decomposition Method
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/02_Implementing_a_Decomposition_Method

from __future__ import division
from __future__ import print_function




import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

#	Create the data
x_vals = np.linspace(start=0, stop=10, num=100)
y_vals = x_vals + np.random.normal(0, 1, 100)

#	Create design matrix
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

#	Create y matrix
y = np.transpose(np.matrix(y_vals))

#	Create tensors
A_tensor = tf.constant(A)
y_tensor = tf.constant(y)

#	Find Cholesky Decomposition
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tA_A)


'''
Original:  Ax=y
			A'Ax = A'y
			LL'x = A'y
			(1) Lz  = A'y
			(2) L'x = z
'''

#	Solve L*y = t(A)*b
tA_y = tf.matmul(tf.transpose(A_tensor), y)
sol1 = tf.matrix_solve(L, tA_y)

#	Solve L'*y = sol1
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

solution_eval = sess.run(sol2)

#	Extract coefficients
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('slope      : ' + str(slope))
print('y_intercept: ' + str(y_intercept))

#	Get best fit line
best_fit = []
for i in x_vals:
	best_fit.append(slope*i + y_intercept)

#	Plot the results
plt.plot(x_vals, y_vals,   'o',  label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
#plt.show()


'''
2018-03-01 11:53:35.265435: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 0.990433446814
y_intercept: 0.1240366284
[Finished in 19.7s]
'''




#-------------------------
# My
#-------------------------

ops.reset_default_graph()
sess = tf.Session()

num_samples = 100
num_feature = 4
num_labels  = 3

# Create data
x_vals = np.random.rand(num_samples, num_feature)
##x_vals = x_vals * 10
y_vals = np.random.randint(num_labels, size=num_samples)
y_vals = np.array(y_vals, dtype=np.float32)

# Create design matrix
ones_column = np.mat(np.repeat(1., num_samples)).T 
A_x = np.concatenate((x_vals, ones_column), axis=1)  #x in A

# Create y
y   = np.array(np.mat(y_vals).T)

# Create tensors
Ax_tensor = tf.constant(A_x, dtype=tf.float32)		#[num_samples, num_feature+1]
y__tensor = tf.constant(y  , dtype=tf.float32)		#[num_samples, 1 			]

'''
Xc = y         X= x in A
X'Xc = X'y
LL'c = X'y
(1) Lz  = X'y
(2) L'c = z
'''

# Solve: Cholesky decomposition
tX_tensor = tf.transpose(Ax_tensor) 			#[num_feature+1, num_samples]
tX_X = tf.matmul(tX_tensor, Ax_tensor)			#[num_feature+1, num_feature+1]
L    = tf.cholesky(tX_X)						#[num_feature+1, ?]

print("Type: \n\t", tX_tensor.dtype, tX_X.dtype, L.dtype)
tX_y = tf.matmul(tX_tensor, y__tensor) 			#[num_feature+1, 1]
'''
2th
2018-03-01 13:48:02.445509: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 0.983158744955
y_intercept: -0.0135999013398
Traceback (most recent call last):
	ValueError: Tensor("Const_1:0", shape=(100, 1), dtype=float64) must be from the same graph as Tensor("transpose:0", shape=(5, 100), dtype=float64).
[Finished in 31.2s with exit code 1]
'''
sol1 = tf.matrix_solve(L, tX_y) 				#[?, 1]
sol2 = tf.matrix_solve(tf.transpose(L), sol1) 	#[num_feature,1] #previously, wrong
#												#[?, num_feature+1]x[.,.] = [?, 1]
#												#			[num_feature+1, 1]

# Extract coefficients
coef = sess.run(sol2)
print("Coefficients and bias:")
print("\t y = ", end="")
for i in range(num_feature):
	print(str(coef[i][0]) + " * Xfeat_" + str(i+1) + " + ", end="")
print(coef[num_feature][0])
print("Cholesky Decomposition:")
print("\t L = ", sess.run(L))

# Get best fit line
ys_eval = tf.matmul(Ax_tensor, sol2) 	#[num_samples, num_feature+1]x[num_feature+1, 1] -> [num_samples, 1]
ys_eval = tf.squeeze(ys_eval)			#[num_samples,]
best_fit = sess.run(ys_eval)

# Plot the results
plt.figure()
plt.plot(x_vals[:,0], y_vals,   'ro', label='Data')
plt.plot(x_vals[:,0], best_fit, 'b*', label='Best fit points')
plt.legend(loc='upper left')
#plt.show()


'''
3th
2018-03-01 13:54:42.080583: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 0.995319515416
y_intercept: 0.012516591807
<dtype: 'float32'> <dtype: 'float32'> <dtype: 'float32'>
Coefficients and bias:
	 y = 0.179987 * Xfeat_1 + 0.169327 * Xfeat_2 + 0.00684631 * Xfeat_3 + 0.0674028 * Xfeat_4 + 0.837441
Cholesky Decomposition:
	 L =  [[ 5.57395697  0.          0.          0.          0.        ]
 [ 3.94336939  4.01089764  0.          0.          0.        ]
 [ 3.88999915  2.33069992  3.1842041   0.          0.        ]
 [ 4.06931591  1.81383073  0.77233684  3.41549778  0.        ]
 [ 8.41600895  3.85599923  1.9377048   1.58861387  2.83260775]]
[Finished in 41.1s]
'''


plt.close('all')

plt.figure()
plt.clf()
all_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for i in range(num_labels):
	this_index = (y_vals == i)
	this_x = x_vals[this_index]
	this_y = y_vals[this_index]
	this_color  = all_color[i] + 'o'
	this_legend = "Class " + str(i)
	plt.plot(this_x[:,0], this_y, this_color, label=this_legend)
	print("Plot", i, "legend", this_legend, "color", this_color)
plt.plot(x_vals[:,0], best_fit, 'k*', label='Best fit points')
plt.legend()#loc='upper left')
plt.show()


plt.close('all')
plt.figure() #plt.clf()
for i in range(num_labels):
	this_index = (y_vals == i)
	this_x = x_vals[this_index]
	this_y = y_vals[this_index]
	this_color  = all_color[i] + 'o'
	this_legend = "Class " + str(i)
	plt.plot(this_x[:,0], this_y, this_color, label=this_legend)

	this_z = best_fit[this_index]
	this_color  = all_color[i] + '*'
	this_legend = 'Best ' + str(i)
	plt.plot(this_x[:,0], this_z, this_color, label=this_legend)
#plt.plot(x_vals, best_fit, 'k*', label='Best fit points')
plt.legend(loc='upper left')
plt.show()

'''
2018-03-01 14:07:35.257570: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 0.942151796048
y_intercept: 0.388437223863
Type: 
	 <dtype: 'float32'> <dtype: 'float32'> <dtype: 'float32'>
Coefficients and bias:
	 y = -0.175296 * Xfeat_1 + -0.115587 * Xfeat_2 + 0.525142 * Xfeat_3 + 0.0209469 * Xfeat_4 + 0.820973
Cholesky Decomposition:
	 L =  [[ 5.65695429  0.          0.          0.          0.        ]
 [ 4.15369797  3.77461433  0.          0.          0.        ]
 [ 4.71401644  1.45486605  3.50402927  0.          0.        ]
 [ 4.6736412   1.96428323  0.84324574  3.43096328  0.        ]
 [ 8.6741991   3.2581439   2.16966605  1.58783054  2.62946916]]
[Finished in 86.1s]
'''



