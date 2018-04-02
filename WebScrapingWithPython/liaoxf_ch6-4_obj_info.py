# coding: utf-8
# OOP




from __future__ import division
from __future__ import print_function




#-------------------
# 获取对象信息
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431866385235335917b66049448ab14a499afd5b24db000
#-------------------


# 当我们拿到一个对象的引用时，如何知道这个对象是什么类型、有哪些方法呢？

## 使用type()
# 首先，我们来判断对象类型，使用type()函数：
# 基本类型都可以用type()判断：

print(type(123))
print(type('str'))
print(type(None))

'''
<type 'int'>
<type 'str'>
<type 'NoneType'>
[Finished in 0.0s]
'''


# 如果一个变量指向函数或者类，也可以用type()判断：

class Animal(object):
	def run(self):
		print('Animal is running...')

a = Animal()

print(type(abs))
print(type(a))

'''
<type 'builtin_function_or_method'>
<class '__main__.Animal'>
[Finished in 0.0s]
'''


# 但是type()函数返回的是什么类型呢？它返回对应的Class类型。如果我们要在if语句中判断，就需要比较两个变量的type类型是否相同：

print(type(123) == type(456))
print(type(123) == int)
print(type('abc') == type('123'))
print(type('abc') == str)
print(type('123') == type(123))

'''
True
True
True
True
False
[Finished in 0.0s]
'''


# 判断基本数据类型可以直接写int，str等，但如果要判断一个对象是否是函数怎么办？可以使用types模块中定义的常量：

import types
def fn():
	pass

print(type(fn) == types.FunctionType)
print(type(abs) == types.BuiltinFunctionType)
print(type(lambda x: x) == types.LambdaType)
print(type((x for x in range(10))) == types.GeneratorType)
print( (x for x in range(10)) )
print( [x for x in range(10)] )

'''
True
True
True
True
<generator object <genexpr> at 0x7f58a44d5730>
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Finished in 0.0s]
'''




## 使用isinstance()
# 对于class的继承关系来说，使用type()就很不方便。我们要判断class的类型，可以使用isinstance()函数。
# 我们回顾上次的例子，如果继承关系是：
# 		object -> Animal -> Dog -> Husky

class Animal(object):
	"""docstring for Animal"""
	pass

class Dog(Animal):
	"""docstring for Dog"""
	pass

class Husky(Dog):
	"""docstring for Husky"""
	pass

a = Animal()
d = Dog()
h = Husky()

print(isinstance(h, Husky), isinstance(h, Dog), isinstance(h, Animal))
print(isinstance(d, Husky), isinstance(d, Dog), isinstance(d, Animal))
print(isinstance(a, Husky), isinstance(a, Dog), isinstance(a, Animal))

'''
True True True
False True True
False False True
[Finished in 0.0s]
'''


# isinstance()判断的是一个对象是否是该类型本身，或者位于该类型的父继承链上。
# 能用type()判断的基本类型也可以用isinstance()判断：

print(isinstance('a', str))
print(isinstance(123, int))
print(isinstance(b'a', bytes))

'''
True
True
True
[Finished in 0.0s]
'''


# 并且还可以判断一个变量是否是某些类型中的一种，比如下面的代码就可以判断是否是list或者tuple：

print(isinstance([1,2,3], (list, tuple) ))
print(isinstance((1,2,3), (list, tuple) )) 

'''
True
True
[Finished in 0.1s]
'''

# 总是优先使用isinstance()判断类型，可以将指定类型及其子类“一网打尽”。





## 使用dir()
# 如果要获得一个对象的所有属性和方法，可以使用dir()函数，它返回一个包含字符串的list，比如，获得一个str对象的所有属性和方法：

print(dir('ABC'))

'''
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__init__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_formatter_field_name_split', '_formatter_parser', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
[Finished in 0.0s]
'''


# 类似__xxx__的属性和方法在Python中都是有特殊用途的，比如__len__方法返回长度。在Python中，如果你调用len()函数试图获取一个对象的长度，实际上，在len()函数内部，它自动去调用该对象的__len__()方法，所以，下面的代码是等价的：

print(len('ABC'))
print('ABC'.__len__())

'''
3
3
[Finished in 0.1s]
'''



class MyDog(object):
	"""docstring for MyDog"""
	def __len__(self):
		return 100

dog = MyDog()
print(len(dog))
'''
100
[Finished in 0.0s]
'''


# 剩下的都是普通属性或方法，比如lower()返回小写的字符串：
print('ABC'.lower())
# 仅仅把属性和方法列出来是不够的，配合getattr()、setattr()以及hasattr()，我们可以直接操作一个对象的状态：

class MyObject(object):
	"""docstring for MyObject"""
	def __init__(self):
		self.x = 9
	def power(self):
		return self.x * self.x 

obj = MyObject()

# 紧接着，可以测试该对象的属性：
print(hasattr(obj, 'x')) 	# 有属性'x'吗？
print(obj.x)
print(hasattr(obj, 'y')) 	# 有属性'y'吗？
print(setattr(obj, 'y', 19)) 	# 设置一个属性'y'
print(hasattr(obj, 'y')) 	# 有属性'y'吗？
print(getattr(obj, 'y')) 	# 获取属性'y'
print(obj.y) 				# 获取属性'y'

'''
abc
True
9
False
None
True
19
19
[Finished in 0.1s]
'''


# 如果试图获取不存在的属性，会抛出AttributeError的错误
# print(getattr(obj, 'z')) 	# 获取属性'z'

'''
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch6-4_obj_info.py", line 237, in <module>
    print(getattr(obj, 'z')) 	# 获取属性'z'
AttributeError: 'MyObject' object has no attribute 'z'
[Finished in 0.0s with exit code 1]
[shell_cmd: python -u "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch6-4_obj_info.py"]
'''


# 可以传入一个default参数，如果属性不存在，就返回默认值：
print(getattr(obj, 'z', 404))
# 获取属性'z'，如果不存在，返回默认值404

# 也可以获得对象的方法：

print(hasattr(obj, 'power')) 	# 有属性'power'吗？
print(getattr(obj, 'power')) 	# 获取属性'power'
fn = getattr(obj, 'power') 		# 获取属性'power'并赋值到变量fn
print(fn) 	# fn指向obj.power
print(fn()) 	# 调用fn()与调用obj.power()是一样的

'''
404
True
<bound method MyObject.power of <__main__.MyObject object at 0x7f03b69b9b50>>
<bound method MyObject.power of <__main__.MyObject object at 0x7f03b69b9b50>>
81
[Finished in 0.1s]
'''




'''
小结
通过内置的一系列函数，我们可以对任意一个Python对象进行剖析，拿到其内部的数据。要注意的是，只有在不知道对象信息的时候，我们才会去获取对象信息。
如果可以直接写：
sum = obj.x + obj.y
就不要写：
sum = getattr(obj, 'x') + getattr(obj, 'y')

一个正确的用法的例子如下：

def readImage(fp):
    if hasattr(fp, 'read'):
        return readData(fp)
    return None
    
假设我们希望从文件流fp中读取图像，我们首先要判断该fp对象是否存在read方法，如果存在，则该对象是一个流，如果不存在，则无法读取。hasattr()就派上了用场。

请注意，在Python这类动态语言中，根据鸭子类型，有read()方法，不代表该fp对象就是一个文件流，它也可能是网络流，也可能是内存中的一个字节流，但只要read()方法返回的是有效的图像数据，就不影响读取图像的功能。
'''



