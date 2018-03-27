# coding: utf-8
# TFGAN
# ref: https://www.jianshu.com/p/08abd788d598
# ref: https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_tensorflow.py




import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim   = 784
z_dim   = 64
h_dim   = 128
lr      = 1e-3
d_steps = 3

#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
if os.path.exists('/home/ubuntu/'):
	mnist = input_data.read_data_sets('/home/ubuntu/Datasets/MNIST', one_hot=True)
	'''
	Extracting /home/ubuntu/Datasets/MNIST/train-images-idx3-ubyte.gz
	Extracting /home/ubuntu/Datasets/MNIST/train-labels-idx1-ubyte.gz
	Extracting /home/ubuntu/Datasets/MNIST/t10k-images-idx3-ubyte.gz
	Extracting /home/ubuntu/Datasets/MNIST/t10k-labels-idx1-ubyte.gz
	[Finished in 5.3s]
	'''
elif os.path.exists('/home/byj/'):
	mnist = input_data.read_data_sets('/home/byj/Datasets/MNIST', one_hot=True)
	#tf.device('/gpu:0')
else:
	raise UserWarning("Please check the path of MNIST.")




if 'session' in locals() and session is not None:
	print('Close interactive session')
	session.close()

'''
tf.device('/gpu:0')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#sess = tf.Session()
'''

'''
tf.device('/cpu:0')
sess = tf.Session()
'''




def plot(samples):
	fig = plt.figure(figsize=(4, 4))
	gs  = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

	return fig 


def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)


def log(x):
	return tf.log(x + 1e-8)



'''
X = tf.placeholder(tf.float32, shape=[None, X_dim]) #shape: [None, X_dim]
z = tf.placeholder(tf.float32, shape=[None, z_dim]) #shape: [None, z_dim]


D_W1 = tf.Variable(xavier_init([X_dim, h_dim])) 	#shape: [X_dim, h_dim]
D_b1 = tf.Variable(tf.zeros(shape=[h_dim])) 		#shape: [h_dim,]
D_W2 = tf.Variable(xavier_init([h_dim, 1])) 		#shape: [h_dim, 1]
D_b2 = tf.Variable(tf.zeros(shape=[1])) 			#shape: [1,]

G_W1 = tf.Variable(xavier_init([z_dim, h_dim])) 	#shape: [z_dim, h_dim]
G_b1 = tf.Variable(tf.zeros(shape=[h_dim])) 		#shape: [h_dim,]
G_W2 = tf.Variable(xavier_init([h_dim, X_dim])) 	#shape: [h_dim, X_dim]
G_b2 = tf.Variable(tf.zeros(shape=[X_dim])) 		#shape: [X_dim]

theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]
'''

with tf.device('/gpu:0'):
	X = tf.placeholder(tf.float32, shape=[None, X_dim]) #shape: [None, X_dim]
	z = tf.placeholder(tf.float32, shape=[None, z_dim]) #shape: [None, z_dim]

	D_W1 = tf.Variable(xavier_init([X_dim, h_dim])) 	#shape: [X_dim, h_dim]
	D_b1 = tf.Variable(tf.zeros(shape=[h_dim])) 		#shape: [h_dim,]
	D_W2 = tf.Variable(xavier_init([h_dim, 1])) 		#shape: [h_dim, 1]
	D_b2 = tf.Variable(tf.zeros(shape=[1])) 			#shape: [1,]

	G_W1 = tf.Variable(xavier_init([z_dim, h_dim])) 	#shape: [z_dim, h_dim]
	G_b1 = tf.Variable(tf.zeros(shape=[h_dim])) 		#shape: [h_dim,]
	G_W2 = tf.Variable(xavier_init([h_dim, X_dim])) 	#shape: [h_dim, X_dim]
	G_b2 = tf.Variable(tf.zeros(shape=[X_dim])) 		#shape: [X_dim]

	theta_G = [G_W1, G_W2, G_b1, G_b2]
	theta_D = [D_W1, D_W2, D_b1, D_b2]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))




def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n]) 	#shape: [m, n]


def generator(z):
	G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1) 		# [None, h_dim]
	G_log_prob = tf.matmul(G_h1, G_W2) + G_b2 			# [None, X_dim]
	G_prob = tf.nn.sigmoid(G_log_prob) 					# [None, X_dim]
	return G_prob


def discriminator(x):
	D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1) 		# [None, h_dim]
	out  = tf.nn.sigmoid(tf.matmul(D_h1, D_W2) + D_b2) 	# [None, 1]
	return out 


G_sample = generator(z)

D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(log(D_real) + log(1 - D_fake))
G_loss = 0.5 * tf.reduce_mean((log(D_fake) - log(1 - D_fake)) ** 2)
'''
original GAN:
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))
'''

D_solver = (tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=theta_G))




#sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
	os.makedirs('out/')

i = 0

for it in range(1000): #1000000):
	X_mb, _ = mnist.train.next_batch(mb_size)
	z_mb    = sample_z(mb_size, z_dim)

	_, D_loss_curr = sess.run(
			[D_solver, D_loss], 
			feed_dict={X: X_mb, z: z_mb}
		)

	_, G_loss_curr = sess.run(
			[G_solver, G_loss], 
			feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
		)


	if  it % 100 ==0: #1000 == 0:
		print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

		samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

		fig = plot(samples)
		plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
		i += 1
		plt.close(fig)
		'''
		>>> '{}.png'.format(str(7).zfill(2))
			'07.png'
		>>> '{}.png'.format(str(7).zfill(3))
			'007.png'
		>>>
		'''










