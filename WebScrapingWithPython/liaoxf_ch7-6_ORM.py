# coding: utf-8
# python 3.5
# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319106919344c4ef8b1e04c48778bb45796e0335839000





##
'''
动态修改有什么意义？直接在MyList定义中写上add()方法不是更简单吗？正常情况下，确实应该直接写，通过metaclass修改纯属变态。

但是，总会遇到需要通过metaclass修改类定义的。ORM就是一个典型的例子。

ORM全称“Object Relational Mapping”，即对象-关系映射，就是把关系数据库的一行映射为一个对象，也就是一个类对应一个表，这样，写代码更简单，不用直接操作SQL语句。

要编写一个ORM框架，所有的类都只能动态定义，因为只有使用者才能根据表的结构定义出对应的类来。

让我们来尝试编写一个ORM框架。

编写底层模块的第一步，就是先把调用接口写出来。比如，使用者如果使用这个ORM框架，想定义一个User类来操作对应的数据库表User，我们期待他写出这样的代码：




其中，父类Model和属性类型StringField、IntegerField是由ORM框架提供的，剩下的魔术方法比如save()全部由metaclass自动完成。虽然metaclass的编写会比较复杂，但ORM的使用者用起来却异常简单。

现在，我们就按上面的接口来实现该ORM。

'''




## 首先来定义Field类，它负责保存数据库表的字段名和字段类型：

class Field(object):

	def __init__(self, name, column_type):
		self.name 		 = name
		self.column_type = column_type

	def __str__(self):
		return '<%s:%s>' % (self.__class__.__name__, self.name)


## 在Field的基础上，进一步定义各种类型的Field，比如StringField，IntegerField等等：

class StringField(Field):
	
	def __init__(self, name):
		super(StringField, self).__init__(name, 'varchar(100)')
	
class IntegerField(Field):

	def __init__(self, name):
		super(IntegerField, self).__init__(name, 'bigint')


## 下一步，就是编写最复杂的ModelMetaclass了：

class ModelMetaclass(type):

	def __new__(cls, name, bases, attrs):
		if name == 'Model':
			return type.__new__(cls, name, bases, attrs)
		print('Found model: %s' % name)
		mappings = dict()
		for k, v in attrs.items():
			if isinstance(v, Field):
				print('\t', end="")
				print('Found mapping: %s ==> %s' % (k, v))
				mappings[k] = v
		for k in mappings.keys():
			attrs.pop(k)
		attrs['__mappings__'] = mappings 	# 保存属性和列的映射关系
		attrs['__table__'] 	  = name 		# 假设表名和类名一致
		return type.__new__(cls, name, bases, attrs)


## 以及基类Model：

class Model(dict, metaclass=ModelMetaclass):

	def __init__(self, **kw):
		super(Model, self).__init__(**kw)

	def __getattr__(self, key):
		try:
			return self[key]
		except:
			raise AttributeError(r"'Model' object has no attribute '%s'" % key)

	def __setattr__(self, key, value):
		self[key] = value

	def save(self):
		fields = []
		params = []
		args   = []
		for k, v in self.__mappings__.items():
			fields.append(v.name)
			params.append('?')
			args.append(  getattr(self, k, None))
		sql = 'insert into %s (%s) values (%s)' % (self.__table__, ','.join(fields), ','.join(params))
		print('SQL: %s' % sql)
		print('ARGS: %s' % str(args))


##
'''
当用户定义一个class User(Model)时，Python解释器首先在当前类User的定义中查找metaclass，如果没有找到，就继续在父类Model中查找metaclass，找到了，就使用Model中定义的metaclass的ModelMetaclass来创建User类，也就是说，metaclass可以隐式地继承到子类，但子类自己却感觉不到。

在ModelMetaclass中，一共做了几件事情：

排除掉对Model类的修改；

在当前类（比如User）中查找定义的类的所有属性，如果找到一个Field属性，就把它保存到一个__mappings__的dict中，同时从类属性中删除该Field属性，否则，容易造成运行时错误（实例的属性会遮盖类的同名属性）；

把表名保存到__table__中，这里简化为表名默认为类名。

在Model类中，就可以定义各种操作数据库的方法，比如save()，delete()，find()，update等等。

我们实现了save()方法，把一个实例保存到数据库中。因为有表名，属性到字段的映射和属性值的集合，就可以构造出INSERT语句。

编写代码试试：
'''





class User(Model):
	"""docstring for User"""
	# 定义类的属性到列的映射：
	id 		 = IntegerField('id')
	name 	 = StringField('username')
	email 	 = StringField('email')
	password = StringField('password')

# 创建一个实例：
u = User(id=41235, name='Hermione Granger', email='test@orm.org', password='my-pwd')
# 创建一个实例：
u.save()



## 输出如下：

'''
(py35env) ubuntu@ubuntu-VirtualBox:~/Program/Code/WebScrapingWithPython$ python liaoxf_ch7-6_ORM.py
Found model: User
	Found mapping: password ==> <StringField:password>
	Found mapping: id ==> <IntegerField:id>
	Found mapping: email ==> <StringField:email>
	Found mapping: name ==> <StringField:username>
SQL: insert into User (email,password,id,username) values (?,?,?,?)
ARGS: ['test@orm.org', 'my-pwd', 41235, 'Hermione Granger']
(py35env) ubuntu@ubuntu-VirtualBox:~/Program/Code/WebScrapingWithPython$ 



Found model: User
	Found mapping: password ==> <StringField :password 	>
	Found mapping: id 		==> <IntegerField:id 		>
	Found mapping: email 	==> <StringField :email 	>
	Found mapping: name 	==> <StringField :username 	>
SQL: insert into User (email,password,id,username) values (?,?,?,?)
ARGS: ['test@orm.org', 'my-pwd', 41235, 'Hermione Granger']

'''



## 可以看到，save()方法已经打印出了可执行的SQL语句，以及参数列表，只需要真正连接到数据库，执行该SQL语句，就可以完成真正的功能。

## 不到100行代码，我们就通过metaclass实现了一个精简的ORM框架，是不是非常简单？


##
'''
小结
metaclass是Python中非常具有魔术性的对象，它可以改变类创建时的行为。这种强大的功能使用起来务必小心。


ref: 
https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319106919344c4ef8b1e04c48778bb45796e0335839000
'''

