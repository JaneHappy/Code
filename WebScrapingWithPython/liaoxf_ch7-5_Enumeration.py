# coding: utf-8
# 使用枚举类
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143191235886950998592cd3e426e91687cdae696e64b000
# 枚举类(Enumeration)


from __future__ import division
from __future__ import print_function




## 当我们需要定义常量时，一个办法是用大写变量通过整数来定义，例如月份：
'''
JAN = 1
FEB = 2
MAR = 3
...
NOV = 11
DEC = 12
'''

Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec = range(1, 12+1)

## 好处是简单，缺点是类型是int，并且仍然是变量。

## 更好的方法是为这样的枚举类型定义一个class类型，然后，每个常量都是class的一个唯一实例。Python提供了Enum类来实现这个功能：

from enum import Enum 

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

## 这样我们就获得了Month类型的枚举类，可以直接使用Month.Jan来引用一个常量，或者枚举它的所有成员：

for name, member in Month.__members__.items():
	print(name, '=>', member, ',', member.value)

'''
Jan => Month.Jan , 1
Feb => Month.Feb , 2
Mar => Month.Mar , 3
Apr => Month.Apr , 4
May => Month.May , 5
Jun => Month.Jun , 6
Jul => Month.Jul , 7
Aug => Month.Aug , 8
Sep => Month.Sep , 9
Oct => Month.Oct , 10
Nov => Month.Nov , 11
Dec => Month.Dec , 12
[Finished in 0.1s]
'''

## value属性则是自动赋给成员的int常量，默认从1开始计数。

## 如果需要更精确地控制枚举类型，可以从Enum派生出自定义类：



from enum import Enum, unique

@unique
class Weekday(Enum):
	Sun = 0 	# Sun的value被设定为0
	Mon = 1
	Tue = 2
	Wed = 3
	Thu = 4
	Fri = 5
	Sat = 6

## @unique装饰器可以帮助我们检查保证没有重复值。

## 访问这些枚举类型可以有若干种方法：

day1 = Weekday.Mon
print(day1)
print(day1 == Weekday.Mon)
print(day1 == Weekday.Tue)
print(day1 == Weekday(1))
print(Weekday.Tue)
print(Weekday['Wed'])
print(Weekday.Thu.value)
print(Weekday(5))
#print(Weekday(7))

'''
Weekday.Mon
True
False
True
Weekday.Tue
Weekday.Wed
4
Weekday.Fri
Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-5_Enumeration.py", line 85, in <module>
    print(Weekday(7))
  File "/usr/local/lib/python2.7/dist-packages/enum/__init__.py", line 347, in __call__
    return cls.__new__(cls, value)
  File "/usr/local/lib/python2.7/dist-packages/enum/__init__.py", line 662, in __new__
    raise ValueError("%s is not a valid %s" % (value, cls.__name__))
ValueError: 7 is not a valid Weekday
[Finished in 0.1s with exit code 1]
[shell_cmd: python -u "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch7-5_Enumeration.py"]
[dir: /home/ubuntu/Program/Code/WebScrapingWithPython]
[path: /home/ubuntu/bin:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin]
'''

print()
for name, member in Weekday.__members__.items():
	print(name, '=>', member)

'''
Sun => Weekday.Sun
Mon => Weekday.Mon
Tue => Weekday.Tue
Wed => Weekday.Wed
Thu => Weekday.Thu
Fri => Weekday.Fri
Sat => Weekday.Sat
[Finished in 0.0s]
'''

## 可见，既可以用成员名称引用枚举常量，又可以直接根据value的值获得枚举常量。






##
'''
练习
把Student的gender属性改造为枚举类型，可以避免使用字符串：
'''

print()

@unique
class Gender(Enum):
	"""docstring for Gender"""
	Male   = 0
	Female = 1

class Student(object):
	def __init__(self, name, gender):
		self.name   = name
		self.gender = gender


# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')


'''

测试通过!
[Finished in 0.1s]
'''



class Student(object):
	def __init__(self, name, gender):
		self.name   = name
		self.gender = Enum('Gender', ('Male', 'Female'))

# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')

'''
测试失败!
[Finished in 0.1s]
'''




##
'''
小结
Enum可以把一组相关常量定义在一个class中，且class不可变，而且成员可以直接比较。
'''




# 将类替换成Enum函数就好
Gender = Enum('gender',('Male','Female'))

class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender


# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')

class Gender(Enum):
    Male = 0
    Female = 1

class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')



'''
测试通过!
测试通过!
[Finished in 0.1s]
'''

