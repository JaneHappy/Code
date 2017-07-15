# coding: utf-8
# chapter 5.1 ~ 5.2

from __future__ import print_function




#====================================
#  chap 5.1    data processing
#====================================

from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集，如果指定地址 /path/to/MNIST_data 下没有已经下载好的数据，那么TensorFlow会自动从表5.1给出的网址下载数据。
# mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("/home/ubuntu/Program/GoogleDL_datasets/MNIST_data/", one_hot=True)

# 打印 Training data size: 55000
print(" Training  data size: ", mnist.train.num_examples)

# 打印 Validating data size: 5000
print("Validating data size: ", mnist.validation.num_examples)

# 打印 Testing data size: 10000
print("  Testing  data size: ", mnist.test.num_examples)

# 打印 Example training data: [0. 0. 0. ... 0.380 0.376 ... 0.]
#print("Example training data:  ", mnist.train.images[0])
##This is sklearn ## print("Example training data:  ", mnist.train.data[0])
# 打印 Example training data label: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print("Example training label: ", mnist.train.labels[0])

'''
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/train-images-idx3-ubyte.gz
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /home/ubuntu/Program/GoogleDL_datasets/MNIST_data/t10k-labels-idx1-ubyte.gz
 Training  data size:  55000
Validating data size:  5000
  Testing  data size:  10000
Example training data:   
Example training label:  [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
[Finished in 18.7s]
'''
print("Training  \t", "images", mnist.train.images.shape, ",\t labels", mnist.train.labels.shape)
print("Validation\t", "images", mnist.validation.images.shape, ",\t labels", mnist.validation.labels.shape)
print("Testing   \t", "images", mnist.test.images.shape, ",\t labels", mnist.test.labels.shape)
print("\t\t\t\t", "images", mnist.train.images[0].shape, ",\t labels", mnist.train.labels[0].shape)
'''
[Finished in 9.6s]
[Finished in 3.5s]
Training  	 images (55000, 784) ,	 labels (55000, 10)
Validation	 images (5000, 784) ,	 labels (5000, 10)
Testing   	 images (10000, 784) ,	 labels (10000, 10)
				 images (784,) ,	 labels (10,)
[Finished in 3.2s]
'''


batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train的集合中选取batch_size个训练数据
print("X shape:", xs.shape)
# 输出 X shape: (100, 784).
print("Y shape:", ys.shape)
# 输出 Y shape: (100, 10).




#====================================
#  chap 5.2    model training
#====================================

