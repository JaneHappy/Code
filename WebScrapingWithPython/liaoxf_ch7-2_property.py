# coding: utf-8

from __future__ import division
from __future__ import print_function



# 使用@property

## 在绑定属性时，如果我们直接把属性暴露出去，虽然写起来很简单，但是，没办法检查参数，导致可以把成绩随便改：
### s = Student()
### s.score = 9999
## 这显然不合逻辑。为了限制score的范围，可以通过一个set_score()方法来设置成绩，再通过一个get_score()来获取成绩，这样，在set_score()方法里，就可以检查参数：

class Student(object):
	"""docstring for Student"""
	
	def get_score(self):
		return self._score

	def set_score(self, value):
		if not isinstance(value, int):
			raise ValueError('score must be an integer!')
		if value < 0 or value > 100:
			raise ValueError('score must be a value between 0 ~ 100!')
		self._score = value

s = Student()
s.set_score(60) 	# ok!
print(s.get_score())
# s.set_score(9999.)
# s.set_score(9999)

'''
60

Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-2_property.py", line 31, in <module>
    s.set_score(9999.)
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-2_property.py", line 23, in set_score
    raise ValueError('score must be an integer!')
ValueError: score must be an integer!
[Finished in 0.0s with exit code 1]

Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-2_property.py", line 32, in <module>
    s.set_score(9999)
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-2_property.py", line 25, in set_score
    raise ValueError('score must be a value between 0 ~ 100!')
ValueError: score must be a value between 0 ~ 100!
[Finished in 0.1s with exit code 1]

'''


## 但是，上面的调用方法又略显复杂，没有直接用属性这么直接简单。

## 有没有既能检查参数，又可以用类似属性这样简单的方式来访问类的变量呢？对于追求完美的Python程序员来说，这是必须要做到的！

## 还记得装饰器（decorator）可以给函数动态加上功能吗？对于类的方法，装饰器一样起作用。Python内置的@property装饰器就是负责把一个方法变成属性调用的：



class Student(object):
	"""docstring for Student"""
	
	@property
	def score(self):
		return self._score

	@score.setter
	def score(self, value):
		if not isinstance(value, int):
			print('ValueError! score must be an integer!')
		if value < 0 or value > 100:
			print('ValueError! score must between 0 ~ 100!')
		self._score = value

## @property的实现比较复杂，我们先考察如何使用。把一个getter方法变成属性，只需要加上@property就可以了，此时，@property本身又创建了另一个装饰器@score.setter，负责把一个setter方法变成属性赋值，于是，我们就拥有一个可控的属性操作：

s = Student()
s.score = 70 	# OK，实际转化为s.set_score(60)
print(s.score) 	# OK，实际转化为s.get_score()
s.score = 99.
s.score = 9999

'''
70
ValueError! score must be an integer!
ValueError! score must between 0 ~ 100!
[Finished in 0.1s]
'''


## 注意到这个神奇的@property，我们在对实例属性操作的时候，就知道该属性很可能不是直接暴露的，而是通过getter和setter方法来实现的。

## 还可以定义只读属性，只定义getter方法，不定义setter方法就是一个只读属性：

class Student(object):
	"""docstring for Student"""
	
	@property
	def birth(self):
		return self._birth

	@birth.setter
	def birth(self, value):
		self._birth = value

	@property
	def age(self):
		return 2015 - self._birth

## 上面的birth是可读写属性，而age就是一个只读属性，因为age可以根据birth和当前时间计算出来。

'''
小结
@property广泛应用在类的定义中，可以让调用者写出简短的代码，同时保证对参数进行必要的检查，这样，程序运行时就减少了出错的可能性。

练习
请利用@property给一个Screen对象加上width和height属性，以及一个只读属性resolution：
'''




class Screen(object):
	"""docstring for Screen"""
	
	@property
	def width(self):
		return self.__width

	@property
	def height(self):
		return self.__height

	@property
	def resolution(self):
		return self.__height * self.__width

	@width.setter
	def width(self, value):
		self.__width = value

	@height.setter
	def height(self, value):
		self.__height = value


# 测试:
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
	print('测试通过!')
else:
	print('测试失败!')


'''
60
70
ValueError! score must be an integer!
ValueError! score must between 0 ~ 100!

resolution = 786432
测试通过!
[Finished in 0.0s]
'''


