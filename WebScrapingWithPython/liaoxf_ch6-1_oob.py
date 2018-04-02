# coding: utf-8
# Object Oriented Programming, OOP




from __future__ import division
from __future__ import print_function




# 面向过程的程序可以用一个dict表示

std1 = {'name':'Michael', 'score':98}
std2 = {'name':'Bob', 'score':81}
def print_score(std):
	print('%s: %s' % (std['name'], std['score']))

print_score(std1)
print_score(std2)
'''
Michael: 98
Bob: 81
[Finished in 0.1s]
'''




class Student(object):
	"""docstring for Student"""
	def __init__(self, name, score):
		self.name  = name
		self.score = score

	def print_score(self):
		print('%11s: %3s' % (self.name, self.score))

# 给对象发消息实际上就是调用对象对应的关联函数，我们称之为对象的方法（Method）。面向对象的程序写出来就像这样

bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
bart.print_score()
lisa.print_score()

'''
Bart Simpson:  59
Lisa Simpson:  87
[Finished in 0.1s]
'''

# 面向对象的设计思想是从自然界中来的，因为在自然界中，类（Class）和实例（Instance）的概念是很自然的。Class是一种抽象概念，比如我们定义的Class——Student，是指学生这个概念，而实例（Instance）则是一个个具体的Student，比如，Bart Simpson和Lisa Simpson是两个具体的Student。
# 所以，面向对象的设计思想是抽象出Class，根据Class创建Instance。
# 面向对象的抽象程度又比函数要高，因为一个Class既包含数据，又包含操作数据的方法。
# 小结 	数据封装、继承和多态是面向对象的三大特点，我们后面会详细讲解。




#-------------------------
# 类和实例
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431864715651c99511036d884cf1b399e65ae0d27f7e000
#-------------------------

class Student(object):
	pass

bart = Student()
print(bart)
print(Student)

'''
<__main__.Student object at 0x7f25d30a3510>
<class '__main__.Student'>
[Finished in 0.0s]
'''

bart.name = 'Bart Malfoy'
print(bart.name)

'''
<__main__.Student object at 0x7fcc4d9b8510>
<class '__main__.Student'>
Bart Malfoy
[Finished in 0.1s]
'''


# 面向对象编程的一个重要特点就是数据封装。在上面的Student类中，每个实例就拥有各自的name和score这些数据。我们可以通过函数来访问这些数据，比如打印一个学生的成绩
# 但是，既然Student实例本身就拥有这些数据，要访问这些数据，就没有必要从外面的函数去访问，可以直接在Student类的内部定义访问数据的函数，这样，就把“数据”给封装起来了。这些封装数据的函数是和Student类本身是关联起来的，我们称之为类的方法：
# 要定义一个方法，除了第一个参数是self外，其他和普通函数一样。要调用一个方法，只需要在实例变量上直接调用，除了self不用传递，其他参数正常传入：
# 这样一来，我们从外部看Student类，就只需要知道，创建实例需要给出name和score，而如何打印，都是在Student类的内部定义的，这些数据和逻辑被“封装”起来了，调用很容易，但却不用知道内部实现的细节。
# 封装的另一个好处是可以给Student类增加新的方法，比如get_grade：

class Student(object):
	"""docstring for Student"""
	def __init__(self, name, score):
		self.name  = name
		self.score = score

	def print_score(self):
		print('%17s: %3s' % (self.name, self.score))

	def get_grade(self):
		if self.score >= 90:
			return 'A'
		elif self.score >= 60:
			return 'B'
		else:
			return 'C'

lisa = Student('Lily Evans', 99)
bart = Student('James Potter', 59)
temp = Student('Sirius Black', 78)
print('%s' % ('-'*7))
#print(lisa.name, lisa.get_grade(), lisa.print_score() )
#print(bart.name, bart.get_grade(), bart.print_score() )
#print(temp.name, temp.get_grade(), temp.print_score() )

#print('name', lisa.name, '\t|\tget_grade()', lisa.get_grade(), '\t|\tprint_score()', lisa.print_score() )
#print('name', bart.name, '\t|\tget_grade()', bart.get_grade(), '\t|\tprint_score()', bart.print_score() )
#print('name', temp.name, '\t|\tget_grade()', temp.get_grade(), '\t|\tprint_score()', temp.print_score() )

print('name', lisa.name, '\t|\tget_grade()', lisa.get_grade(), '\t|\tprint_score() ', end=""); lisa.print_score()
print('name', bart.name, '\t|\tget_grade()', bart.get_grade(), '\t|\tprint_score() ', end=""); bart.print_score()
print('name', temp.name, '\t|\tget_grade()', temp.get_grade(), '\t|\tprint_score() ', end=""); temp.print_score()

'''
<__main__.Student object at 0x7f7133798690>
<class '__main__.Student'>
Bart Malfoy
       Lily Evans:  99
Lily Evans A None
     James Potter:  59
James Potter C None
     Sirius Black:  78
Sirius Black B None
[Finished in 0.1s]

<__main__.Student object at 0x7f29a622f690>
<class '__main__.Student'>
Bart Malfoy
-------
       Lily Evans:  99 			Lily Evans A None
     James Potter:  59 			James Potter C None
     Sirius Black:  78 			Sirius Black B None
[Finished in 0.0s]

<__main__.Student object at 0x7fda47594690>
<class '__main__.Student'>
Bart Malfoy
-------
       Lily Evans:  99
name Lily Evans 	|	get_grade() A 	|	print_score() None
     James Potter:  59
name James Potter 	|	get_grade() C 	|	print_score() None
     Sirius Black:  78
name Sirius Black 	|	get_grade() B 	|	print_score() None
[Finished in 0.1s]

<__main__.Student object at 0x7f2c515f4690>
<class '__main__.Student'>
Bart Malfoy
-------
name Lily Evans 	|	get_grade() A 	|	print_score()        Lily Evans:  99
name James Potter 	|	get_grade() C 	|	print_score()      James Potter:  59
name Sirius Black 	|	get_grade() B 	|	print_score()      Sirius Black:  78
[Finished in 0.1s]
'''

bart.age = 8
print(bart.age)
#print(lisa.age)

'''
Michael: 98
Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch6-1_oob.py", line 174, in <module>
    print(lisa.age)
AttributeError: 'Student' object has no attribute 'age'
Bob: 81
Bart Simpson:  59
Lisa Simpson:  87
<__main__.Student object at 0x7fa1bec85690>
<class '__main__.Student'>
Bart Malfoy
-------
name Lily Evans 	|	get_grade() A 	|	print_score()        Lily Evans:  99
name James Potter 	|	get_grade() C 	|	print_score()      James Potter:  59
name Sirius Black 	|	get_grade() B 	|	print_score()      Sirius Black:  78
8
[Finished in 0.1s with exit code 1]
[shell_cmd: python -u "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch6-1_oob.py"]
[dir: /home/ubuntu/Program/Code/WebScrapingWithPython]
[path: /home/ubuntu/bin:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin]
'''



#-------------------------
#-------------------------




#-------------------------
#-------------------------






