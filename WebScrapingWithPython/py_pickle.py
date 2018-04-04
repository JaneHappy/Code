#!/usr/bin/env python3

import sys
print(sys.version)

'''
2.7.12+ (default, Sep 17 2016, 12:08:02) 
[GCC 6.2.0 20160914]
[Finished in 0.4s]

(py35env) ubuntu@ubuntu-VirtualBox:~/Program/Code$ python WebScrapingWithPython/py_pickle.py
3.5.2+ (default, Sep 22 2016, 12:18:14) 
[GCC 6.2.0 20160927]
(py35env) ubuntu@ubuntu-VirtualBox:~/Program/Code$

>>> import platform
>>> platform.python_version()
'3.5.2+'
>>> import sys
>>> sys.version
'3.5.2+ (default, Sep 22 2016, 12:18:14) \n[GCC 6.2.0 20160927]'
>>> sys.version_info
sys.version_info(major=3, minor=5, micro=2, releaselevel='final', serial=0)

'''




#使用pickle模块将数据对象保存到文件


import pickle

data1 = {'a': [1, 2.0, 3, 4+6j],
		 'b': ('string', u'Unicode string'),
		 'c': None}

selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)

'''
>>> a = [1,2,3]
>>> a.append(a)
>>> a
[1, 2, 3, [...]]
>>> a[3]
[1, 2, 3, [...]]
>>> a[-1]
[1, 2, 3, [...]]
>>> 

>>> a  = [1,2,3]
>>> a
[1, 2, 3]
>>> a.append(4)
>>> a
[1, 2, 3, 4]
>>> a.append(a)
>>> a
[1, 2, 3, 4, [...]]
>>> a[-1]
[1, 2, 3, 4, [...]]
>>> len(a)
5
>>> len(a[-1])
5
>>> len(a[-1][-1])
5
>>> len(a[-1][-1][-1])
5
>>> 
'''

output = open('data.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(data1, output)

# Pickle the list using the highest protocol available.
pickle.dump(selfref_list, output, -1)


output.close()




#使用pickle模块从文件中重构python对象

'''
>>> import pprint, pickle
>>> pkl_file = open('data.pkl', 'rb')
>>> data1 = pickle.load(pkl_file)
>>> data1
{'b': ('string', 'Unicode string'), 'a': [1, 2.0, 3, (4+6j)], 'c': None}
>>> pprint.pprint(data1)
{'a': [1, 2.0, 3, (4+6j)], 'b': ('string', 'Unicode string'), 'c': None}
>>> print(data1)
{'b': ('string', 'Unicode string'), 'a': [1, 2.0, 3, (4+6j)], 'c': None}
>>> 
'''

'''
>>> pprint.pprint(d1)
{'a': [1, 2.0, 3, (4+6j)], 'b': ('string', 'Unicode string'), 'c': None}
>>> pprint.pprint(d2)
[1, 2, 3, <Recursion on list with id=140554294889672>]
>>> 
'''

import pprint, pickle

pkl_file = open('data.pkl', 'rb')

data2 = pickle.load(pkl_file)
pprint.pprint(data2)

data3 = pickle.load(pkl_file)
pprint.pprint(data3)

pkl_file.close()


'''
(py35env) ubuntu@ubuntu-VirtualBox:~/Program/Code$ python WebScrapingWithPython/py_pickle.py
3.5.2+ (default, Sep 22 2016, 12:18:14) 
[GCC 6.2.0 20160927]
{'a': [1, 2.0, 3, (4+6j)], 'b': ('string', 'Unicode string'), 'c': None}
[1, 2, 3, <Recursion on list with id=139851890819848>]
(py35env) ubuntu@ubuntu-VirtualBox:~/Program/Code$ 

'''







