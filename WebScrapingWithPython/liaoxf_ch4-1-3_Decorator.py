# coding: utf-8
# 装饰器
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318435599930270c0381a3b44db991cd6d858064ac0000

from __future__ import division
from __future__ import print_function




## 由于函数也是一个对象，而且函数对象可以被赋值给变量，所以，通过变量也能调用该函数。

def now():
	print('2015-3-25')

f = now
f()

'''
2015-3-25
[Finished in 0.0s]
'''

## 函数对象有一个__name__属性，可以拿到函数的名字：

print(now.__name__)
print(f.__name__)

'''
now
now
[Finished in 0.1s]
'''


## 现在，假设我们要增强now()函数的功能，比如，在函数调用前后自动打印日志，但又不希望修改now()函数的定义，这种在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）。

## 本质上，decorator就是一个返回函数的高阶函数。所以，我们要定义一个能打印日志的decorator，可以定义如下：

def log(func):
	def wrapper(*args, **kw):
		print('call %s():' % func.__name__)
		return func(*args, **kw)
	return wrapper

## 观察上面的log，因为它是一个decorator，所以接受一个函数作为参数，并返回一个函数。我们要借助Python的@语法，把decorator置于函数的定义处：

@log
def now():
	print('2016-03-25')

print('--')
now()
print(now.__name__)

'''
--
call now():
2016-03-25
[Finished in 0.1s]

call now():
2016-03-25
wrapper
'''

##
'''
把@log放到now()函数的定义处，相当于执行了语句：
now = log(now)

由于log()是一个decorator，返回一个函数，所以，原来的now()函数仍然存在，只是现在同名的now变量指向了新的函数，于是调用now()将执行新函数，即在log()函数中返回的wrapper()函数。

wrapper()函数的参数定义是(*args, **kw)，因此，wrapper()函数可以接受任意参数的调用。在wrapper()函数内，首先打印日志，再紧接着调用原始函数。

如果decorator本身需要传入参数，那就需要编写一个返回decorator的高阶函数，写出来会更复杂。比如，要自定义log的文本：
'''



def log(text):
	def decorator(func):
		def wrapper(*args, **kw):
			print('%s %s():' % (text, func.__name__))
			return func(*args, **kw)
		return wrapper
	return decorator

## 这个3层嵌套的decorator用法如下：
@log('execute')
def now():
	print('2017-04-29')

print()
now()
print(now.__name__)

'''
execute now():
2017-04-29
[Finished in 0.1s]

execute now():
2017-04-29
wrapper
[Finished in 0.0s]
'''

##
'''
和两层嵌套的decorator相比，3层嵌套的效果是这样的：

>>> now = log('execute')(now)

我们来剖析上面的语句，首先执行log('execute')，返回的是decorator函数，再调用返回的函数，参数是now函数，返回值最终是wrapper函数。

以上两种decorator的定义都没有问题，但还差最后一步。因为我们讲了函数也是对象，它有__name__等属性，但你去看经过decorator装饰之后的函数，它们的__name__已经从原来的'now'变成了'wrapper'：

>>> now.__name__
'wrapper'

因为返回的那个wrapper()函数名字就是'wrapper'，所以，需要把原始函数的__name__等属性复制到wrapper()函数中，否则，有些依赖函数签名的代码执行就会出错。

不需要编写wrapper.__name__ = func.__name__这样的代码，Python内置的functools.wraps就是干这个事的，所以，一个完整的decorator的写法如下：
'''



print('\n')
print('functools --------')


import functools

def log(func):
	@functools.wraps(func)
	def wrapper(*args, **kw):
		print('call %s():' % func.__name__)
		return func(*args, **kw)
	return wrapper

@log
def now():
	print('2018/4/2')

print()
now()
print(now.__name__)
'''
functools --------
execute now():
2017-04-29
wrapper
[Finished in 0.0s]



functools --------

call now():
2018/4/2
now
[Finished in 0.1s]
'''


## 或者针对带参数的decorator：

import functools

def log(text):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kw):
			print('%s %s():' % (text, func.__name__))
			return func(*args, **kw)
		return wrapper
	return decorator

@log('test')
def now():
	print("2018/4/1 April Fool's Day")

print()
now()
print(now.__name__)

'''


functools --------

call now():
2018/4/2
now

test now():
2018/4/1 April Fool's Day
now
[Finished in 0.0s]
'''


##
'''
import functools是导入functools模块。模块的概念稍候讲解。现在，只需记住在定义wrapper()的前面加上@functools.wraps(func)即可。

练习
请设计一个decorator，它可作用于任何函数上，并打印该函数的执行时间：
'''







print('\n\nExample\n')
import functools
import time

'''
def metric(fn):
	print('%s executed in %s ms' % (fn.__name__, 10.24))
	return fn

Example

fast executed in 10.24 ms
slow executed in 10.24 ms
[Finished in 0.2s]
'''

'''
def metric(fn):
	@functools.wraps(fn)
	def wrapper(*args, **kw):
		print('%s executed in %s ms' % (fn.__name__, fn(*args, **kw) ))
		return fn(*args, **kw)
	return wrapper

Example

fast executed in 33 ms
slow executed in 7986 ms
[Finished in 0.3s]
'''


def metric(fn):
	@functools.wraps(fn)
	def printTime(*args, **kw):
		t0 = time.time()
		result = fn(*args, **kw)
		t1 = time.time()
		print('%s executed in %f ms' % (fn.__name__, (t1-t0)*1000. ))
		return result #fn(*args, **kw)
	return printTime

'''
Example

fast executed in 2.065897 ms
slow executed in 123.679876 ms
[Finished in 0.2s]

time.time()返回的是秒
'''


# 测试
@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;

@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

f = fast(11, 22)
s = slow(11, 22, 33)
if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')







##
'''
小结
在面向对象（OOP）的设计模式中，decorator被称为装饰模式。OOP的装饰模式需要通过继承和组合来实现，而Python除了能支持OOP的decorator外，直接从语法层次支持decorator。Python的decorator可以用函数实现，也可以用类实现。

decorator可以增强函数的功能，定义起来虽然有点复杂，但使用起来非常灵活和方便。

请编写一个decorator，能在函数调用的前后打印出'begin call'和'end call'的日志。

再思考一下能否写出一个@log的decorator，使它既支持：
'''


import functools

def log(text='call'):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kw):
			print('begin %s \t %s():' % (text, func.__name__ ))
			answer = func(*args, **kw)
			print('end %s \t %s():' % (  text, func.__name__ ))
			return answer
		return wrapper
	return decorator

print('\n\nbegin and end call\n')

@log()
def f():
	print('2018/4/3')

f()
print(f.__name__)

@log('execute')
def f():
	print('2018/4/4')

f()
print(f.__name__)

'''
begin and end call

begin call 	 f():
2018/4/3
end call 	 f():
f
begin execute 	 f():
2018/4/4
end execute 	 f():
f
[Finished in 0.2s]
'''




import functools

def log(*text):
	#if text == None:
	#	text = 'call'
	if len(text) == 0:
		text = 'call'
	else:
		text = text[0]
	print("log", text)

	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kw):
			print('\t', 'begin %s \t %s():' % (text, func.__name__ ))
			answer = func(*args, **kw)
			print('\t', 'end %s \t %s():' % (  text, func.__name__ ))
			return answer
		return wrapper
	return decorator

print('\n\nsecond begin and end call\n')

@log()
def f():
	print('\t\t', '2018/4/3')

f()
print(f.__name__)

@log('execute')
def f():
	print('\t\t', '2018/4/4')

f()
print(f.__name__)



'''
https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431752945034eb82ac80a3e64b9bb4929b16eeed1eb9000
可变参数
在Python函数中，还可以定义可变参数。顾名思义，可变参数就是传入的参数个数是可变的，可以是1个、2个到任意个，还可以是0个。

>>> def log(*text):
...     print(text)
... 
>>> log()
()
>>> log(1)
(1,)
>>> log('test')
('test',)
>>> 

'''



