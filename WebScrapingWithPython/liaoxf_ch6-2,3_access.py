# coding: utf-8
# OOP




from __future__ import division
from __future__ import print_function




#-------------------
# 访问限制
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318650247930b1b21d7d3c64fe38c4b5a80d4469ad7000
#-------------------


# 在Class内部，可以有属性和方法，而外部代码可以通过直接调用实例变量的方法来操作数据，这样，就隐藏了内部的复杂逻辑。
# 但是，从前面Student类的定义来看，外部代码还是可以自由地修改一个实例的name、score属性：
# 如果要让内部属性不被外部访问，可以把属性的名称前加上两个下划线__，在Python中，实例的变量名如果以__开头，就变成了一个私有变量（private），只有内部可以访问，外部不能访问，所以，我们把Student类改一改：


class Student(object):
	"""docstring for Student"""
	def __init__(self, name, score):
		self.__name  = name
		self.__score = score

	def print_score(self):
		print('%17s: %4s' % (self.__name, self.__score))

bart = Student('Bart Simpson', 59)
#print(bart.__name)
'''
Traceback (most recent call last):
  File "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch6-2_access.py", line 34, in <module>
    print(bart.__name)
AttributeError: 'Student' object has no attribute '__name'
[Finished in 0.1s with exit code 1]
[shell_cmd: python -u "/home/ubuntu/Program/Code/WebScrapingWithPython/liaoxf_ch6-2_access.py"]
[dir: /home/ubuntu/Program/Code/WebScrapingWithPython]
[path: /home/ubuntu/bin:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin]
'''

# 改完后，对于外部代码来说，没什么变动，但是已经无法从外部访问实例变量.__name和实例变量.__score了：
# 这样就确保了外部代码不能随意修改对象内部的状态，这样通过访问限制的保护，代码更加健壮。

# 但是如果外部代码要获取name和score怎么办？可以给Student类增加get_name和get_score这样的方法：

# 如果又要允许外部代码修改score怎么办？可以再给Student类增加set_score方法：


class Student(object):
	"""docstring for Student"""
	def __init__(self, name, score):
		self.__name  = name
		self.__score = score

	def print_score(self):
		return('%17s: %4s' % (self.__name, self.__score))

	def get_name( self):
		return self.__name
	def get_score(self):
		return self.__score

	def set_score(self, score):
		#if 0 <= score <= 100:
		# >>> 0<= score<= 100 		False
		if 0<= score and score<=100:
			self.__score = score
		else:
			raise ValueError('bad score')


# 需要注意的是，在Python中，变量名类似__xxx__的，也就是以双下划线开头，并且以双下划线结尾的，是特殊变量，特殊变量是可以直接访问的，不是private变量，所以，不能用__name__、__score__这样的变量名。

# 有些时候，你会看到以一个下划线开头的实例变量名，比如_name，这样的实例变量外部是可以访问的，但是，按照约定俗成的规定，当你看到这样的变量时，意思就是，“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”。

# 双下划线开头的实例变量是不是一定不能从外部访问呢？其实也不是。不能直接访问__name是因为Python解释器对外把__name变量改成了_Student__name，所以，仍然可以通过_Student__name来访问__name变量：

# print(bart._Student__name)
'''
Bart Simpson
[Finished in 0.0s]
'''




#-------------------
# 继承和多态
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431865288798deef438d865e4c2985acff7e9fad15e3000
#-------------------


# 在OOP程序设计中，当我们定义一个class的时候，可以从某个现有的class继承，新的class称为子类（Subclass），而被继承的class称为基类、父类或超类（Base class、Super class）。


class Animal(object):
	"""docstring for Animal"""
	def run(self):
		print('Animal is running...')

class Dog(Animal):
	"""docstring for Dog"""
	pass

class Cat(Animal):
	"""docstring for Cat"""
	pass


# 继承有什么好处？最大的好处是子类获得了父类的全部功能。由于Animial实现了run()方法，因此，Dog和Cat作为它的子类，什么事也没干，就自动拥有了run()方法：

Herny  = Dog()
Herny.run()
Violet = Cat()
Violet.run()

'''
Animal is running...
Animal is running...
[Finished in 0.0s]
'''


# 继承的第二个好处需要我们对代码做一点改进。你看到了，无论是Dog还是Cat，它们run()的时候，显示的都是Animal is running...，符合逻辑的做法是分别显示Dog is running...和Cat is running...，因此，对Dog和Cat类改进如下：

class Dog(Animal):
	def run(self):
		print('Dog is running...')
	def eat(self):
		print('Eating meat...')

class Cat(Animal):
	def run(self):
		print('Cat is running...')

Herny.run()
Violet.run()
Herny  = Dog()
Herny.run()
Violet = Cat()
Violet.run()
'''
Animal is running...
Animal is running...
Dog is running...
Cat is running...
[Finished in 0.1s]
'''


# 当子类和父类都存在相同的run()方法时，我们说，子类的run()覆盖了父类的run()，在代码运行的时候，总是会调用子类的run()。这样，我们就获得了继承的另一个好处：多态。

# 要理解什么是多态，我们首先要对数据类型再作一点说明。当我们定义一个class的时候，我们实际上就定义了一种数据类型。我们定义的数据类型和Python自带的数据类型，比如str、list、dict没什么两样：


a = list() 		# a是list类型
b = Animal() 	# b是Animal类型
c = Dog() 		# c是Dog类型

# 判断一个变量是否是某个类型可以用isinstance()判断：
print( isinstance(a, list) )
print( isinstance(b, Animal) )
print( isinstance(c, Dog) )

print(isinstance(Herny , Dog), isinstance(Herny , Animal))
print(isinstance(Violet, Cat), isinstance(Violet, Animal))

'''
True
True
True
True True
True True
[Finished in 0.0s]
'''

# 看来a、b、c确实对应着list、Animal、Dog这3种类型。
# 但是等等，试试：
print( isinstance(c, Animal) )
'''
True
[Finished in 0.0s]
'''

d = Animal()
print(isinstance(d, Animal), isinstance(d, Dog), isinstance(d, Cat))
'''
True False False
[Finished in 0.1s]
'''



# 要理解多态的好处，我们还需要再编写一个函数，这个函数接受一个Animal类型的变量：

def run_twice(animal):
	animal.run()
	animal.run()

run_twice(Animal())
run_twice(Dog())
run_twice(Cat())

'''
Animal is running...
Animal is running...
Dog is running...
Dog is running...
Cat is running...
Cat is running...
[Finished in 0.1s]
'''

# 看上去没啥意思，但是仔细想想，现在，如果我们再定义一个Tortoise类型，也从Animal派生：

class Tortoise(Animal):
	"""docstring for Tortoise"""
	def run(self):
		print('Tortoise is running slowly...')

run_twice(Tortoise())
'''
Tortoise is running slowly...
Tortoise is running slowly...
[Finished in 0.1s]
'''


# 你会发现，新增一个 Animal 的子类，不必对 run_twice() 做任何修改，实际上，任何依赖 Animal 作为参数的函数或者方法都可以不加修改地正常运行，原因就在于多态。

# 多态的好处就是，当我们需要传入 Dog、Cat、Tortoise…… 时，我们只需要接收 Animal 类型就可以了，因为 Dog、Cat、Tortoise…… 都是 Animal 类型，然后，按照 Animal 类型进行操作即可。由于 Animal 类型有 run() 方法，因此，传入的任意类型，只要是 Animal 类或者子类，就会自动调用实际类型的 run() 方法，这就是多态的意思：

# 对于一个变量，我们只需要知道它是 Animal类型，无需确切地知道它的子类型，就可以放心地调用 run() 方法，而具体调用的 run() 方法是作用在 Animal、Dog、Cat 还是 Tortoise 对象上，由运行时该对象的确切类型决定，这就是多态真正的威力：调用方只管调用，不管细节，而当我们新增一种 Animal 的子类时，只要确保 run()方法编写正确，不用管原来的代码是如何调用的。这就是著名的“开闭”原则：

# 对扩展开放：允许新增 Animal 子类；

# 对修改封闭：不需要修改依赖 Animal 类型的 run_twice() 等函数。

# 继承还可以一级一级地继承下来，就好比从爷爷到爸爸、再到儿子这样的关系。而任何类，最终都可以追溯到根类 object，这些继承关系看上去就像一颗倒着的树。比如如下的继承树：



'''
静态语言 vs 动态语言
对于静态语言（例如Java）来说，如果需要传入Animal类型，则传入的对象必须是Animal类型或者它的子类，否则，将无法调用run()方法。

对于Python这样的动态语言来说，则不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了：

class Timer(object):
    def run(self):
        print('Start...')
这就是动态语言的“鸭子类型”，它并不要求严格的继承体系，一个对象只要“看起来像鸭子，走起路来像鸭子”，那它就可以被看做是鸭子。

Python的“file-like object“就是一种鸭子类型。对真正的文件对象，它有一个read()方法，返回其内容。但是，许多对象，只要有read()方法，都被视为“file-like object“。许多函数接收的参数就是“file-like object“，你不一定要传入真正的文件对象，完全可以传入任何实现了read()方法的对象。

小结
继承可以把父类的所有功能都直接拿过来，这样就不必重零做起，子类只需要新增自己特有的方法，也可以把父类不适合的方法覆盖重写。

动态语言的鸭子类型特点决定了继承不像静态语言那样是必须的。
'''





