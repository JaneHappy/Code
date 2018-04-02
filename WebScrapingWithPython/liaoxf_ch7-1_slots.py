# coding: utf-8
'''
面向对象高级编程

数据封装、继承和多态只是面向对象程序设计中最基础的3个概念。在Python中，面向对象还有很多高级特性，允许我们写出非常强大的功能。
我们会讨论多重继承、定制类、元类等概念。
'''




from __future__ import division
from __future__ import print_function




# 使用__slots__
# 阅读: 225437

## 正常情况下，当我们定义了一个class，创建了一个class的实例后，我们可以给该实例绑定任何属性和方法，这就是动态语言的灵活性。先定义class：

class Student(object):
	"""docstring for Student"""
	pass

## 然后，尝试给实例绑定一个属性：

s = Student()
s.name = 'Michael' 	# 动态给实例绑定一个属性
print(s.name)
'''
Michael
[Finished in 0.0s]
'''

## 还可以尝试给实例绑定一个方法：

def set_age(self, age): 	# 定义一个函数作为实例方法
	self.age = age

from types import MethodType
s.set_age = MethodType(set_age, s) 	# 给实例绑定一个方法
s.set_age(25) 	# 调用实例方法
print(s.age) 	# 测试结果

'''
25
[Finished in 0.0s]
'''


## 但是，给一个实例绑定的方法，对另一个实例是不起作用的：

s2 = Student() 	# 创建新的实例
#s2.set_age(25) 	# 尝试调用方法

'''
Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-1_slots.py", line 56, in <module>
    s2.set_age(25) 	# 尝试调用方法
AttributeError: 'Student' object has no attribute 'set_age'
[Finished in 0.1s with exit code 1]
'''


## 为了给所有实例都绑定方法，可以给class绑定方法：

def set_score(self, score):
	self.score = score

Student.set_score = set_score

## 给class绑定方法后，所有实例均可调用：

s.set_score(100)
print(s.score)
s2.set_score(99)
print(s2.score)

'''
100
99
[Finished in 0.0s]
'''

## 通常情况下，上面的set_score方法可以直接定义在class中，但动态绑定允许我们在程序运行的过程中动态给class加上功能，这在静态语言中很难实现。




#-------------------

# 使用__slots__

## 但是，如果我们想要限制实例的属性怎么办？比如，只允许对Student实例添加name和age属性。

## 为了达到限制的目的，Python允许在定义class的时候，定义一个特殊的__slots__变量，来限制该class实例能添加的属性：

class Student(object):
	"""docstring for Student"""
	__slots__ = ('name', 'age') 	# 用tuple定义允许绑定的属性名称

s = Student() 		# 创建新的实例
s.name = 'Michael' 	# 绑定属性'name'
s.age  = 25 		# 绑定属性'age'
#s.score = 99 		# 绑定属性'score'
'''
Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-1_slots.py", line 107, in <module>
    s.score = 99 		# 绑定属性'score'
AttributeError: 'Student' object has no attribute 'score'
[Finished in 0.1s with exit code 1]
'''

## 由于'score'没有被放到__slots__中，所以不能绑定score属性，试图绑定score将得到AttributeError的错误。

## 使用__slots__要注意，__slots__定义的属性仅对当前类实例起作用，对继承的子类是不起作用的：


class GraduateStudent(Student):
	"""docstring for GraduateStudent"""
	pass

g = GraduateStudent()
g.score = 9999
'''
[Finished in 0.0s]
'''

## 除非在子类中也定义__slots__，这样，子类实例允许定义的属性就是自身的__slots__加上父类的__slots__。




'''
大家看完后可能会有两个疑问：
slots限制的属性包不包含方法？
slots限制属性后，类能不能动态绑定属性？
我试了下，直接给出结论吧：前者包含，后者能继续绑定



大家的MethodType有三个参数吗？
为什么我查阅网上的资料，很多都有三个参数的
from types import MethodType 
def set_age(self, arg):
    self.age = arg
class Student(object):
    pass
s_one = Student()
s_one.set_age = MethodType(set_age,s_one,Student)
例如这个代码，可以绑定在类内绑定方法，是不是python版本不同，还是我操作有误，编译器一直提示只能有2个参数。

自问自答
前面如果用类名称，就是绑定类里
前面如果用方法，就是绑定在方法里



1, 限制实例中属性的名字
	__slot__=(touple)
	只允许存在touple里的属性
2, slot不能继承
	子类中再定义会自动加入父类的限制




'''

