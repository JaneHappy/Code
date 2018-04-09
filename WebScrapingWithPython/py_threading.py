# coding: utf-8

from __future__ import division
from __future__ import print_function




#=============
# 全局解释锁（GIL）
#=============

import time
import threading

def profile(func):
	def wrapper(*args, **kwargs):
		import time
		start = time.time()
		result = func(*args, **kwargs)
		end   = time.time()
		print('COST: {}'.format(end - start))
		return result
	return wrapper

def fib(n):
	if n <= 2:
		return 1
	return fib(n-1) + fib(n-2)

@profile
def nothread():
	fib(35)
	fib(35)

@profile
def hasthread():
	for i in range(2):
		t = threading.Thread(target=fib, args=(35,))
		t.start()
	main_thread = threading.currentThread()
	for t in threading.enumerate():
		if t is main_thread:
			continue
		t.join()


'''
ubuntu@ubuntu-VirtualBox:~/Program/Code$ python
Python 2.7.12+ (default, Sep 17 2016, 12:08:02) 
[GCC 6.2.0 20160914] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import time
>>> import threading
>>> threading.currentThread()
<_MainThread(MainThread, started 139928943601408)>
>>> threading.enumerate()
[<_MainThread(MainThread, started 139928943601408)>]
>>> threading.Thread(target=abs, args=(-35,))
<Thread(Thread-1, initial)>
>>> len(threading.enumerate())
1
>>> 
'''


# nothread()
# hasthread()

'''
COST: 4.88517308235
COST: 7.23335409164
[Finished in 12.2s]
'''




#=============
# 同步机制
#=============
# Semaphore（信号量）

'''
import time
from random import random
from threading import Thread, Semaphore

sema = Semaphore(3)

def foo(tid):
	with sema:
		print('{} acquire sema.'.format(tid))
		wt = random() * 2
		time.sleep(wt)
	print('{} release sema'.format(tid))

threads = []

for i in range(5):
	t = Thread(target=foo, args=(i,))
	threads.append(t)
	t.start()

for t in threads:
	t.join()

'''


'''
0 acquire sema.
1 acquire sema.
2 acquire sema.
1 release sema
3 acquire sema.
2 release sema
4 acquire sema.
0 release sema
4 release sema
3 release sema
[Finished in 1.6s]

0,1,2
0,2,3
0,3,4
3,4
3,
.
'''




#=============
# Lock（锁）
# 互斥锁，其实相当于信号量为1
#=============

'''
import time
from threading import Thread

value = 0

def getlock():
	global value
	new = value + 1
	time.sleep(0.001) # 使用sleep让线程有机会切换
	value = new

threads = []

for i in range(100):
	t = Thread(target=getlock)
	t.start()
	threads.append(t)

for t in threads:
	t.join()

print(value)



34
[Finished in 0.2s]
'''



import time
from threading import Thread, Lock 

value = 0
lock = Lock()

def getlock():
	global value
	with lock:
		new = value + 1
		time.sleep(0.001)
		value = new

threads = []

for i in range(100):
	t = Thread(target=getlock)
	t.start()
	threads.append(t)

for t in threads:
	t.join()

print(value)


'''
100
[Finished in 0.2s]
'''



#=============
#=============




#=============
'''
Ref:

http://www.dongwm.com/archives/%E4%BD%BF%E7%94%A8Python%E8%BF%9B%E8%A1%8C%E5%B9%B6%E5%8F%91%E7%BC%96%E7%A8%8B-%E7%BA%BF%E7%A8%8B%E7%AF%87/

'''
