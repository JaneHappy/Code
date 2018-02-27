# coding: utf-8
# tensorflow_cookbook
#	01 Introduction
#		06 Implementing Activation Functions
#	https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/06_Implementing_Activation_Functions/06_activation_functions.py




import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops 	## [Finished in 17.6s]
ops.reset_default_graph()

# Open a graph session
sess = tf.Session()

# X range 
x_vals = np.linspace(start=-10, stop=10, num=100)


# Activation Functions 
y_relu 		= sess.run(tf.nn.relu(	  x_vals))	# ReLU activation
y_relu6 	= sess.run(tf.nn.relu6(	  x_vals))	# ReLU-6 activation
y_sigmoid 	= sess.run(tf.nn.sigmoid( x_vals))	# Sigmoid activation
y_tanh 		= sess.run(tf.nn.tanh(	  x_vals))	# Hyper Tangent activation
y_softsign 	= sess.run(tf.nn.softsign(x_vals))	# Softsign activation
y_softplus 	= sess.run(tf.nn.softplus(x_vals))	# Softplus activation
y_elu 		= sess.run(tf.nn.elu(	  x_vals))	# Exponential linear activation

t = [-1, 0, 1.]
print(sess.run( tf.nn.relu(		t) ))
print(sess.run( tf.nn.relu6(	t) ))
print(sess.run( tf.nn.sigmoid(	t) ))
print(sess.run( tf.nn.tanh(		t) ))
print(sess.run( tf.nn.softsign(	t) ))
print(sess.run( tf.nn.softplus(	t) ))
print(sess.run( tf.nn.elu(		t) ))
#print(sess.run( tf.nn.(t) ))


# Plot the different functions
plt.figure()

plt.plot(x_vals, y_softplus , 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu 	, 'b:' , label='ReLU'	 , linewidth=2)
plt.plot(x_vals, y_relu6 	, 'g-.', label='ReLU-6'	 , linewidth=2)
plt.plot(x_vals, y_elu 		, 'k-' , label='ExpLU'	 , linewidth=2)
#plt.ylim([-1.5, 7])
plt.plot(x_vals, y_sigmoid	, 'y--', label='Softsign', linewidth=2)
plt.plot(x_vals, y_tanh 	, 'm:' , label='Tanh'	 , linewidth=2)
plt.plot(x_vals, y_softsign	, 'mp:', label='Softsign', linewidth=2)
#plt.ylim([-2,   2])

plt.ylim([-7, 7])
## plt.legend(loc='top left')  	#error!
plt.legend(loc='upper left')
plt.show()




'''
2018-02-27 20:00:08.123418: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
[ 0.  0.  1.]
[ 0.  0.  1.]
[ 0.26894143  0.5         0.7310586 ]
[-0.76159418  0.          0.76159418]
[-0.5  0.   0.5]
[ 0.31326166  0.69314718  1.31326163]
[-0.63212055  0.          1.        ]
/usr/local/lib/python2.7/dist-packages/matplotlib/legend.py:326: UserWarning: Unrecognized location "top left". Falling back on "best"; valid locations are
	right
	center left
	upper right
	lower right
	best
	center
	lower left
	center right
	upper left
	upper center
	lower center

  six.iterkeys(self.codes))))

'''




plt.figure(figsize=(12,5))  #(15,8)
''' or
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)
'''

plt.subplot(121)
plt.plot(x_vals, y_softplus , 'r--', label='Softplus', linewidth=2) #4)#2)
plt.plot(x_vals, y_relu 	, 'b:' , label='ReLU'	 , linewidth=4) #3)#4)
plt.plot(x_vals, y_relu6 	, 'g-.', label='ReLU-6'	 , linewidth=2) #2)#2)
plt.plot(x_vals, y_elu 		, 'k-' , label='ExpLU'	 , linewidth=1) #1)#1)
plt.ylim([-1.5, 7])
plt.legend(loc='upper left')

plt.subplot(122)
plt.plot(x_vals, y_sigmoid	, 'r--', label='Softsign', linewidth=2)
plt.plot(x_vals, y_tanh 	, 'b:' , label='Tanh'	 , linewidth=2)
plt.plot(x_vals, y_softsign	, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,   2])
plt.legend(loc='upper left')

plt.plot([-10,10], [ 1, 1], 'k-', linewidth=1)
plt.plot([-10,10], [-1,-1], 'k-', linewidth=1)

plt.show()




