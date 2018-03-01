# coding: utf-8
# tensorflow cookbook
#	03 Linear Regression
#		01 Using the Matrix Inverse Method
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/01_Using_the_Matrix_Inverse_Method/01_lin_reg_inverse.ipynb

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
y_vals = x_vals + np.random.normal(0, 1, 100)  #mean=0,stdev=1

#	Create design matrix
x_vals_column = np.transpose(np.matrix(x_vals))  #np.mat(x_vals)
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))  #np.mat(np.repeat()).T
A = np.column_stack((x_vals_column, ones_column))  #np.concatenate((one_column,x_vals_column), axis=1)

#	Format the y matrix
y = np.transpose(np.matrix(y_vals))

#	Create tensors
A_tensor = tf.constant(A)
y_tensor = tf.constant(y)

#	Matrix inverse solution
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, y_tensor)

solution_eval = sess.run(solution)
#	Extract coefficients
slope  = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('slope      : ' + str(slope))
print('y_intercept: ' + str(y_intercept))
#	Get best fit line
best_fit = []
for i in x_vals:
	best_fit.append(slope*i + y_intercept)

#	Plot the results
plt.figure()
plt.plot(x_vals, y_vals,   'o',  label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()


'''
2018-03-01 10:33:18.595087: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 1.01298253366
y_intercept: -0.0924192720918
[Finished in 30.3s]
'''




#-------------------------
#	My 3d Plot
#-------------------------

ops.reset_default_graph()
sess = tf.Session()

num_samples = 100
num_feature = 3 
x_vals = np.random.rand(num_samples, num_feature)
y_vals = np.random.randint(2, size=num_samples)  #size=(num_samles,) #size=(num_samples)

# to matrix
one_column = np.mat(np.repeat(1., num_samples)).T
A = np.concatenate((x_vals, np.array(one_column) ), axis=1)

y_column   = np.array(np.mat(y_vals).T)

# to tensor
A_tensor = tf.constant(A)
y_tensor = tf.constant(y)

tA_A     = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product  = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, y_tensor)

solution_eval = sess.run(solution)

# get coefficients
print("Coefficients and Bias:")
print("\t y = ", end="")
for i in range(num_feature):
	print(str(solution_eval[i][0]) + " `x_i"+str(i+1) + " + ", end="")
print(str(solution_eval[num_feature][0]))

# best fit line
ys_eval = tf.matmul(A_tensor, solution)  #Ax, [num_samples, num_feature+1]x[num_feature+1,1] -> [num_samples,1]
ys_eval = tf.squeeze(ys_eval)  #[num_samples,]
best_fit = sess.run(ys_eval)

# 3d Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
#x1_3d, x2_3d = np.meshgrid(x_vals[:,0], x_vals[:,1])  #x_3d = x_vals.T[0]
#ax.plot_surface(x1_3d, x2_3d, best_fit, rstride=1, cstride=1, cmap='rainbow')

ax.scatter(x_vals[:,0], x_vals[:,1], y_vals, c='m')

ax.scatter(x_vals[:,0], x_vals[:,1], best_fit, c='y')
ax.set_zlabel('Label (y)')
ax.set_ylabel('Feature (x1)')
ax.set_xlabel('Feature (x2)')

plt.show()




'''
2th?
2018-03-01 10:53:33.206273: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 1.04583894836
y_intercept: -0.22075130573
Coefficients and Bias:
	 y = 0.332112523923 `x_i1 + 1.29963350012 `x_i2 + 4.18587012087
[Finished in 9.3s]


4th
2018-03-01 11:09:06.148471: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 0.955581867537
y_intercept: 0.12527586033
Coefficients and Bias:
	 y = 0.227080713909 `x_i1 + -0.8786790866 `x_i2 + 5.2283409543
[Finished in 42.4s]


5th
2018-03-01 11:12:19.560968: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 1.04596766283
y_intercept: -0.202235749085
Coefficients and Bias:
	 y = 1.1673993739 `x_i1 + 1.36167177963 `x_i2 + 3.78926043185
[Finished in 59.0s]

--------------- previously, num_feature = 2
--------------- after ,     num_feature = 3

6th
2018-03-01 11:15:42.491723: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
slope      : 0.979625109938
y_intercept: 0.115425758141
Coefficients and Bias:
	 y = 1.20492395523 `x_i1 + 1.85352517055 `x_i2 + -1.70861222984 `x_i3 + 4.45721597004
[Finished in 80.8s]

'''






## worse solution
'''
worse_solu = tf.matmul(tf.matrix_inverse(A_tensor), y_tensor)
worse_eval = sess.run(worse_solu)

#worse_solu:  error!
ValueError: Dimensions must be equal, but are 100 and 3 for 'MatrixInverse_1' (op: 'MatrixInverse') with input shapes: [100,3].
'''




