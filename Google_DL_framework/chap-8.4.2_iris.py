# -*- coding: utf-8 -*-
#  chapter 8.4.2    时间序列预测




from __future__ import print_function


# 为了方便数据处理，本程序使用了 sklearn 工具包，关于这个工具包更多的信息可以参考 http://scikit-learn.org/。
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf 


# 导入 TFLearn。
learn = tf.contrib.learn


# 自定义模型，对于给定的输入数据（features）以及其对应的正确答案（target），返回在这些输入上的预测值、损失值以及训练步骤。
def my_model(features, target):
	# 将预测的目标转换为 one-hot 编码的形式，因为共有三个类别，所以向量长度为3. 经过转化后，第一个类别表示为 (1,0,0)，第二个为 (0,1,0)，第三个为 (0,0,1)。
	target = tf.one_hot(target, 3, 1, 0)

	# 定义模型以及其在给定数据上的损失函数。TFLearn 通过 logistic_regression 封装了一个单层全连接神经网络。
	logits, loss = learn.models.logistic_regression(features, target)

	# 创建模型的优化器，并得到优化步骤。
	train_op = tf.contrib.layers.optimize_loss(
		loss, 										# 损失函数
		tf.contrib.framework.get_global_step(), 	# 获取训练步数并在训练时更新
		optimizer='Adagrad', 						# 定义优化器
		learning_rate=0.1) 							# 定义学习率

	# 返回在给定数据上的预测结果、损失值以及优化步骤。
	return tf.arg_max(logits, 1), loss, train_op


# 加载 iris 数据集，并划分为训练集合和测试集合。
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# 对自定义的模型进行封装。
classifier = learn.Estimator(model_fn=my_model)

# 使用封装好的模型和训练数据执行 100 轮迭代。
classifier.fit(x_train, y_train, steps=100)

# 使用训练好的模型进行结果预测。
y_predicted = classifier.predict(x_test)

# 计算模型的准确度。
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.9f%%' % (score * 100.))

# 运行以上程序可以得到输出：




'''
tf.one_hot

one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
'''




'''
/usr/local/lib/python2.7/dist-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpaPJZGI
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:48: calling fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with y is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:48: calling fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/models.py:173: softmax_classifier (from tensorflow.contrib.learn.python.learn.ops.losses_ops) is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Use `tf.contrib.losses.softmax_cross_entropy` and explicit logits computation.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/ops/losses_ops.py:75: softmax_cross_entropy (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.softmax_cross_entropy instead. Note that the order of the logits and labels arguments has been changed.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:398: compute_weighted_loss (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.compute_weighted_loss instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:151: add_loss (from tensorflow.contrib.framework.python.ops.arg_scope) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.add_loss instead.
2017-07-25 14:37:54.422261: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-25 14:37:54.423764: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-25 14:37:54.424856: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
WARNING:tensorflow:From /home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py:51: calling predict (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/models.py:173: softmax_classifier (from tensorflow.contrib.learn.python.learn.ops.losses_ops) is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Use `tf.contrib.losses.softmax_cross_entropy` and explicit logits computation.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/ops/losses_ops.py:75: softmax_cross_entropy (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.softmax_cross_entropy instead. Note that the order of the logits and labels arguments has been changed.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:398: compute_weighted_loss (from tensorflow.contrib.losses.python.losses.loss_ops) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.compute_weighted_loss instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/losses/python/losses/loss_ops.py:151: add_loss (from tensorflow.contrib.framework.python.ops.arg_scope) is deprecated and will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.add_loss instead.
Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/Google_DL_framework/chap-8.4.2_example.py", line 54, in <module>
    score = metrics.accuracy_score(y_test, y_predicted)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/metrics/classification.py", line 172, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/metrics/classification.py", line 72, in _check_targets
    check_consistent_length(y_true, y_pred)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/utils/validation.py", line 177, in check_consistent_length
    lengths = [_num_samples(X) for X in arrays if X is not None]
  File "/usr/local/lib/python2.7/dist-packages/sklearn/utils/validation.py", line 122, in _num_samples
    type(x))
TypeError: Expected sequence or array-like, got <type 'generator'>
[Finished in 3.2s]
'''




