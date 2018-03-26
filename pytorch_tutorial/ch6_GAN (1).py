# coding: utf-8
# GAN
# ref: https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/tree/master/chapter6_GAN




import os

import torch 
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs 
from torchvision.utils import save_image




# 进行数据预处理和迭代器的构建
im_tfs = tfs.Compose([tfs.ToTensor(), 
					  tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 	# 标准化
					 ])
#train_set  = MNIST('./mnist', transform=im_tfs)
#train_set  = MNIST('./mnist', transform=im_tfs, download=True)

train_set  = MNIST('./torch_mnist', transform=im_tfs, download=True)
#train_set  = MNIST('/home/ubuntu/Datasets/torch_mnist', transform=im_tfs)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)

'''
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Processing...
Done!
[Finished in 94.0s]

[Finished in 4.8s]
[Finished in 0.6s]
'''


# 定义网络
class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(28*28, 128),
			nn.ReLU(True),
			nn.Linear(128, 64), 
			nn.ReLU(True),
			nn.Linear(64, 12), 
			nn.ReLU(True), 
			nn.Linear(12, 3) 	# 输出的 code 是 3 维，便于可视化
		)

		self.decoder = nn.Sequential(
			nn.Linear(3, 12), 
			nn.ReLU(True), 
			nn.Linear(12, 64), 
			nn.ReLU(True), 
			nn.Linear(64, 128), 
			nn.ReLU(True), 
			nn.Linear(128, 28*28), 
			nn.Tanh() 	# 这里定义的编码器和解码器都是 4 层神经网络作为模型，中间使用 relu 激活函数，最后输出的 code 是三维，注意解码器最后我们使用 tanh 作为激活函数，因为输入图片标准化在 -1 ~ 1 之间，所以输出也要在 -1 ~ 1 这个范围内，最后我们可以验证一下
		)

	def forward(self, x):
		encode = self.encoder(x)
		decode = self.decoder(encode)
		return encode, decode 


net = autoencoder()
x = Variable(torch.randn(1, 28*28)) 	# batch size 是 1
code, _ = net(x)
print(code.shape)

'''
(1L, 3L)
[Finished in 0.7s]
'''


criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_img(x):
	'''
	定义一个函数将最后的结果转换回图片
	'''
	x = 0.5 * (x + 1.)
	x = x.clamp(0, 1)
	x = x.view(x.shape[0], 1, 28, 28)
	return x

'''
print(to_img(x))

(1L, 3L)
Variable containing:
(0 ,0 ,.,.) = 
	Columns  0 to  8 
	Columns  9 to 17 
	Columns 18 to 26 
	Columns 27 to 27 
[torch.FloatTensor of size 1x1x28x28]
[Finished in 0.7s]
'''


# 开始训练自动编码器
for e in range(100):
	for im, _ in train_data:
		im = im.view(im.shape[0], -1)
		im = Variable(im)
		# 前向传播
		_, output = net(im)
		loss = criterion(output, im) / im.shape[0] 	# 平均
		# 反向传播
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (e+1) % 20 == 0: 	# 每 20 次，将生成的图片保存一下
		print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data[0] ))
		pic = to_img(output.cpu().data)
		if not os.path.exists('./simple_autoencoder'):
			os.mkdir('./simple_autoencoder')
		save_image(pic, './simple_autoencoder/image_{}.png'.format(e + 1))




