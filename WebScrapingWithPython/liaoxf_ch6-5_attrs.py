# coding: utf-8
# OOP



from __future__ import division
from __future__ import print_function



#----------------------
# 实例属性和类属性
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319117128404c7dd0cf0e3c4d88acc8fe4d2c163625000
#----------------------



# 由于Python是动态语言，根据类创建的实例可以任意绑定属性。

# 给实例绑定属性的方法是通过实例变量，或者通过self变量：

class Student(object):
	"""docstring for Student"""
	def __init__(self, name):
		self.name = name

s = Student('Bob')
s.score = 90
'''
print(s.name, s.score)
Bob 90
'''


# 但是，如果Student类本身需要绑定一个属性呢？可以直接在class中定义属性，这种属性是类属性，归Student类所有：

# 当我们定义了一个类属性后，这个属性虽然归类所有，但类的所有实例都可以访问到。来测试一下：

class Student(object):
	"""docstring for Student"""
	name = 'Student'

s = Student() 	# 创建实例s
print(s.name) 	# 打印name属性，因为实例并没有name属性，所以会继续查找class的name属性
'''
Student
[Finished in 0.0s]
'''

print(Student.name) 	# 打印类的name属性
s.name = 'Michael' 		# 给实例绑定name属性
print(s.name) 			# 由于实例属性优先级比类属性高，因此，它会屏蔽掉类的name属性
print(Student.name) 	# 但是类属性并未消失，用Student.name仍然可以访问
del s.name 				# 如果删除实例的name属性
print(s.name) 			# 再次调用s.name，由于实例的name属性没有找到，类的name属性就显示出来了

'''
Student
Student
Michael
Student
Student
[Finished in 0.0s]
'''


# 从上面的例子可以看出，在编写程序的时候，千万不要对实例属性和类属性使用相同的名字，因为相同名称的实例属性将屏蔽掉类属性，但是当你删除实例属性后，再使用相同的名称，访问到的将是类属性。





# 练习
# 为了统计学生人数，可以给Student类增加一个类属性，每创建一个实例，该属性自动增加：

class Student(object):
	"""docstring for Student"""
	count = 0

	def __init__(self, name):
		self.name = name
		Student.count += 1
		# count += 1 		#bug!

print('First \t', Student.count)

if Student.count != 0:
	print('测试失败!')
else:
	bart = Student('Bart')
	print('Second \t', Student.count)
	if Student.count != 1:
		print('测试失败!')
	else:
		print('Third \t', Student.count)
		lisa = Student('Bart')
		print('Third \t', Student.count)
		if Student.count != 2:
			print('测试失败!')
		else:
			print('Students:', Student.count)
			print('测试通过!')


'''
First 	 0
Second 	 1
Third 	 1
Third 	 2
Students: 2
测试通过!
[Finished in 0.1s]
'''



