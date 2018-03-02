# coding: utf-8
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/08_Learning_Tic_Tac_Toe
# 圈叉游戏（二人轮流在井字形九格中画 O 或 X，先将三个 O 或 X 连成一线者为胜）网络


from __future__ import division
from __future__ import print_function




# Learning Optimal Tic-Tac-Toe Moves via a Neural Network
#---------------------------------------
#
# We will build a one-hidden layer neural network
#  to predict the optimal response given a set
#  of tic-tac-toe boards.
import csv
import random
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops
ops.reset_default_graph()


# X = 1
# O = -1
# empty = 0
# response on 1-9 grid for placement of next '1'


# For example, the 'test_board' is:
#
#   O  |  -  |  -
# -----------------
#   X  |  O  |  O
# -----------------
#   -  |  -  |  X
#
# board above = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
# Optimal response would be position 6, where
# the position numbers are:
#
#   0  |  1  |  2
# -----------------
#   3  |  4  |  5
# -----------------
#   6  |  7  |  8


response = 6
batch_size = 50
symmetry = ['rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h']


# Print a board
def print_board(board):
	symbols = ['O', ' ', 'X']  #[-1,0,1]
	board_plus1 = [int(x)+1  for x in board]
	print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]] + ' | ' + symbols[board_plus1[2]])
	#print('___________')
	print('-----------')
	print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]] + ' | ' + symbols[board_plus1[5]])
	#print('___________')
	print('-----------')
	print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]] + ' | ' + symbols[board_plus1[8]])


## Given a board, a response, and a transformation, get the new board+response
def get_symmetry(board, response, transformation):
	'''
	:param board: list of integers 9 long:
		 opposing mark = -1
		 friendly mark = 1
		 empty space   = 0
	:param transformation: one of five transformations on a board:
		 'rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h'
	:return: tuple: (new_board, new_response)
	'''

	if transformation == 'rotate180':
		new_response = 8 - response
		return(board[::-1], new_response)
		'''
		>>> c = range(9)		[0, 1, 2, 3, 4, 5, 6, 7, 8]
		>>> c[::-1]				[8, 7, 6, 5, 4, 3, 2, 1, 0]
		'''
	elif transformation == 'rotate90':
		new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
		tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
		return([value  for item in tuple_board for value in item], new_response)
		'''
		>>> c1					[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
		>>> zip(*c1)			[(6, 3, 0), (7, 4, 1), (8, 5, 2)]
		'''
	elif transformation == 'rotate270':
		new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
		tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
		return([value  for item in tuple_board for value in item], new_response)
		'''
		>>> c2					[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
		>>>  zip(*c2)			[(0, 3, 6), (1, 4, 7), (2, 5, 8)]
		>>> zip(*c2)[::-1]		[(2, 5, 8), (1, 4, 7), (0, 3, 6)]
		'''
	elif transformation == 'flip_v':
		new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
		return(board[6:9] + board[3:6] + board[0:3], new_response)
		'''
		>>> c 							[0, 1, 2, 3, 4, 5, 6, 7, 8]
		>>> c[6:9]+c[3:6]+c[0:3]		[6, 7, 8, 3, 4, 5, 0, 1, 2]
		'''
	elif transformation == 'flip_h':
		new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
		new_board = board[::-1]
		return(new_board[6:9] + new_board[3:6] + new_board[0:3], new_response)
		'''
		>>> c3 = c[::-1]				[8, 7, 6, 5, 4, 3, 2, 1, 0]
		>>> c3[6:9]+c3[3:6]+c3[0:3]		[2, 1, 0, 5, 4, 3, 8, 7, 6]
		'''
	else:
		raise ValueError('Method not implemented.')

'''
original board: 			 vertical (or horizontal) 垂直（或水平）
#   0  |  1  |  2
# -----------------
#   3  |  4  |  5
# -----------------
#   6  |  7  |  8

'rotate180'				'rotate90' 				'rotate270' 			'flip_v' 				'flip_h'
#   8  |  7  |  6  		#   6  |  3  |  0  		#   2  |  5  |  8  		#   6  |  7  |  8  		#   2  |  1  |  0
# -----------------		# -----------------		# -----------------		# -----------------		# -----------------
#   5  |  4  |  3  		#   7  |  4  |  1  		#   1  |  4  |  7  		#   3  |  4  |  5  		#   5  |  4  |  3  
# -----------------		# -----------------		# -----------------		# -----------------		# -----------------
#   2  |  1  |  0  		#   8  |  5  |  2  		#   0  |  3  |  6  		#   0  |  1  |  2  		#   8  |  7  |  6  

'''


## Read in board move csv file
def get_moves_from_csv(csv_file):
	'''
	:param csv_file: csv file location containing the boards w/ responses
    :return: moves:  list of moves with index of best response
	'''
	moves = []
	with open(csv_file, 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			moves.append( ([int(x) for x in row[0:9]], int(row[9])) )
	return(moves)

'''
>>> import csv
>>> cf = open('base_tic_tac_toe_moves.csv', 'rt')
>>> cf
<open file 'base_tic_tac_toe_moves.csv', mode 'rt' at 0x7f2ba80f1660>
>>> cr = csv.reader(cf, delimiter=',')
>>> cr
<_csv.reader object at 0x7f2ba80ee9f0>

>>> moves = []
>>> for row in cr:
...     moves.append( ([int(x)  for x in row[0:9]], int(row[9])) )
... 
>>> moves
[([0, 0, 0, 0, -1, 0, 0, 0, 0], 0), ([0, -1, 0, 0, 0, 0, 0, 0, 0], 0), ([0, 0, 0, 0, 0, -1, 0, 0, 0], 6), ([-1, 0, 0, 0, 0, 0, 0, 0, 0], 4), ([0, 0, 0, 0, 0, 0, 1, -1, -1], 3), ([0, -1, 0, 0, 1, 0, 0, 0, -1], 0), ([0, -1, 1, 0, 0, -1, 0, 0, 0], 7), ([-1, 0, 0, 0, -1, 0, 0, 0, 1], 6), ([0, 0, 1, 0, 0, -1, -1, 0, 0], 4), ([0, 0, -1, 0, 0, 0, 0, -1, 1], 4), ([1, 0, 0, -1, 0, 0, 0, -1, 0], 2), ([0, 0, -1, 0, 1, 0, -1, 0, 0], 5), ([-1, 0, 0, 1, -1, -1, 0, 0, 1], 6), ([-1, 1, -1, 0, 1, 0, 0, 1, 0], 8), ([0, 0, 0, -1, 0, 1, 1, -1, -1], 1), ([-1, 1, 0, 0, 0, -1, 0, -1, 1], 3), ([0, -1, 1, 0, 1, -1, -1, 0, 0], 8), ([0, 0, -1, 1, 0, -1, 0, -1, 1], 0), ([1, -1, 0, 0, -1, 0, 0, 0, 0], 7), ([1, 0, -1, 0, -1, 0, 0, 0, 0], 6), ([1, 0, 0, 0, -1, 0, -1, 0, 0], 2), ([1, 0, 0, 0, -1, -1, 0, 0, 0], 3), ([1, 0, 0, 0, -1, 0, 0, 0, -1], 6), ([1, -1, 0, -1, -1, 0, 0, 1, 0], 5), ([1, -1, 0, 0, -1, 0, -1, 1, 0], 2), ([1, -1, -1, 0, -1, 0, 0, 1, 0], 6), ([1, -1, 0, 0, -1, -1, 0, 1, 0], 3), ([1, 0, -1, -1, -1, 0, 1, 0, 0], 8), ([1, -1, 1, 0, -1, 0, -1, 0, 0], 7), ([1, 0, 0, 1, -1, -1, -1, 0, 0], 2), ([1, 0, 0, -1, -1, 0, 1, 0, -1], 5)]
>>> moves[0]
([0, 0, 0, 0, -1, 0, 0, 0, 0], 0)

>>> cf.close()
>>> cf = open('base_tic_tac_toe_moves.csv', 'rt')
>>> cr = csv.reader(cf, delimiter=',')
>>> for row in cr:
...     print row
... 
['0', '0', '0', '0', '-1', '0', '0', '0', '0', '0']
['0', '-1', '0', '0', '0', '0', '0', '0', '0', '0']
['0', '0', '0', '0', '0', '-1', '0', '0', '0', '6']
['-1', '0', '0', '0', '0', '0', '0', '0', '0', '4']
['0', '0', '0', '0', '0', '0', '1', '-1', '-1', '3']
['0', '-1', '0', '0', '1', '0', '0', '0', '-1', '0']
['0', '-1', '1', '0', '0', '-1', '0', '0', '0', '7']
['-1', '0', '0', '0', '-1', '0', '0', '0', '1', '6']
['0', '0', '1', '0', '0', '-1', '-1', '0', '0', '4']
['0', '0', '-1', '0', '0', '0', '0', '-1', '1', '4']
['1', '0', '0', '-1', '0', '0', '0', '-1', '0', '2']
['0', '0', '-1', '0', '1', '0', '-1', '0', '0', '5']
['-1', '0', '0', '1', '-1', '-1', '0', '0', '1', '6']
['-1', '1', '-1', '0', '1', '0', '0', '1', '0', '8']
['0', '0', '0', '-1', '0', '1', '1', '-1', '-1', '1']
['-1', '1', '0', '0', '0', '-1', '0', '-1', '1', '3']
['0', '-1', '1', '0', '1', '-1', '-1', '0', '0', '8']
['0', '0', '-1', '1', '0', '-1', '0', '-1', '1', '0']
['1', '-1', '0', '0', '-1', '0', '0', '0', '0', '7']
['1', '0', '-1', '0', '-1', '0', '0', '0', '0', '6']
['1', '0', '0', '0', '-1', '0', '-1', '0', '0', '2']
['1', '0', '0', '0', '-1', '-1', '0', '0', '0', '3']
['1', '0', '0', '0', '-1', '0', '0', '0', '-1', '6']
['1', '-1', '0', '-1', '-1', '0', '0', '1', '0', '5']
['1', '-1', '0', '0', '-1', '0', '-1', '1', '0', '2']
['1', '-1', '-1', '0', '-1', '0', '0', '1', '0', '6']
['1', '-1', '0', '0', '-1', '-1', '0', '1', '0', '3']
['1', '0', '-1', '-1', '-1', '0', '1', '0', '0', '8']
['1', '-1', '1', '0', '-1', '0', '-1', '0', '0', '7']
['1', '0', '0', '1', '-1', '-1', '-1', '0', '0', '2']
['1', '0', '0', '-1', '-1', '0', '1', '0', '-1', '5']

>>> cf.close()
>>> cm = []
>>> cf = open('base_tic_tac_toe_moves.csv', 'rt')
>>> cr = csv.reader(cf, delimiter=',')
>>> for row in cr:
...     cm.append(row)
... 
>>> cm
[['0', '0', '0', '0', '-1', '0', '0', '0', '0', '0'], ['0', '-1', '0', '0', '0', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '-1', '0', '0', '0', '6'], ['-1', '0', '0', '0', '0', '0', '0', '0', '0', '4'], ['0', '0', '0', '0', '0', '0', '1', '-1', '-1', '3'], ['0', '-1', '0', '0', '1', '0', '0', '0', '-1', '0'], ['0', '-1', '1', '0', '0', '-1', '0', '0', '0', '7'], ['-1', '0', '0', '0', '-1', '0', '0', '0', '1', '6'], ['0', '0', '1', '0', '0', '-1', '-1', '0', '0', '4'], ['0', '0', '-1', '0', '0', '0', '0', '-1', '1', '4'], ['1', '0', '0', '-1', '0', '0', '0', '-1', '0', '2'], ['0', '0', '-1', '0', '1', '0', '-1', '0', '0', '5'], ['-1', '0', '0', '1', '-1', '-1', '0', '0', '1', '6'], ['-1', '1', '-1', '0', '1', '0', '0', '1', '0', '8'], ['0', '0', '0', '-1', '0', '1', '1', '-1', '-1', '1'], ['-1', '1', '0', '0', '0', '-1', '0', '-1', '1', '3'], ['0', '-1', '1', '0', '1', '-1', '-1', '0', '0', '8'], ['0', '0', '-1', '1', '0', '-1', '0', '-1', '1', '0'], ['1', '-1', '0', '0', '-1', '0', '0', '0', '0', '7'], ['1', '0', '-1', '0', '-1', '0', '0', '0', '0', '6'], ['1', '0', '0', '0', '-1', '0', '-1', '0', '0', '2'], ['1', '0', '0', '0', '-1', '-1', '0', '0', '0', '3'], ['1', '0', '0', '0', '-1', '0', '0', '0', '-1', '6'], ['1', '-1', '0', '-1', '-1', '0', '0', '1', '0', '5'], ['1', '-1', '0', '0', '-1', '0', '-1', '1', '0', '2'], ['1', '-1', '-1', '0', '-1', '0', '0', '1', '0', '6'], ['1', '-1', '0', '0', '-1', '-1', '0', '1', '0', '3'], ['1', '0', '-1', '-1', '-1', '0', '1', '0', '0', '8'], ['1', '-1', '1', '0', '-1', '0', '-1', '0', '0', '7'], ['1', '0', '0', '1', '-1', '-1', '-1', '0', '0', '2'], ['1', '0', '0', '-1', '-1', '0', '1', '0', '-1', '5']]

>>> a = cm[0]
>>> a
['0', '0', '0', '0', '-1', '0', '0', '0', '0', '0']
>>> a[0:9]
['0', '0', '0', '0', '-1', '0', '0', '0', '0']
>>> a = cm[-1]
>>> a
['1', '0', '0', '-1', '-1', '0', '1', '0', '-1', '5']
>>> a[0:9]
['1', '0', '0', '-1', '-1', '0', '1', '0', '-1']
>>> a[9]
'5'
>>> 
'''


## Get random board with optimal move
def get_rand_move(moves, n=1, rand_transforms=2):
	'''
	:param moves: 			list of the boards w/responses
	:param n: 				how many board positions with responses to return in a list form
	:param rand_transforms: how many random transforms performed on each
	:return: (board, response), board is a list of 9 integers, response is 1 int
	'''
	(board, response) = random.choice(moves)
	possible_transforms = ['rotate90', 'rotate180', 'rotate270', 'flip_v', 'flip_h']
	for i in range(rand_transforms):
		random_transform = random.choice(possible_transforms)
		(board, response) = get_symmetry(board, response, random_transform)
	return(board, response)

'''
>>> import random
>>> random.choice(a)
'5'
>>> random.choice(moves)
([-1, 0, 0, 0, 0, 0, 0, 0, 0], 4)
>>> 
'''


# Get list of optimal moves w/ responses
moves = get_moves_from_csv('base_tic_tac_toe_moves.csv')

# Create a train set:
train_length = 500
train_set    = []
for t in range(train_length):
	train_set.append(get_rand_move(moves))

# To see if the network learns anything new, we will remove
# all instances of the board [-1, 0, 0, 1, -1, -1, 0, 0, 1],
# which the optimal response will be the index '6'.  We will
# Test this at the end.
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set  = [ x  for x in train_set  if x[0] != test_board]


def init_weights(shape):
	return(tf.Variable(tf.random_normal(shape)))

def model(X, A1, A2, bias1, bias2):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), bias1))
	layer2 = tf.add(tf.matmul(layer1, A2), bias2)
	return(layer2) # note that we dont take the softmax at the end because our cost fn does that for us


X = tf.placeholder(dtype=tf.float32, shape=[None, 9])
Y = tf.placeholder(dtype=tf.int32,   shape=[None])

A1    = init_weights([9, 81])
bias1 = init_weights([81])
A2    = init_weights([81, 9])
bias2 = init_weights([9])

model_output = model(X, A1, A2, bias1, bias2)
'''
	layer1 		#[None,81]
	layer2 		#[None, 9]
model_output 	#[None,9]
'''

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=Y))
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
prediction = tf.argmax(model_output, 1)
'''
loss 	  :		#[None,] -> a number
prediction:  index of maximum in every row
				#[None,]
'''

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
for j in range(10000):
	rand_indices = np.random.choice(range(len(train_set)), batch_size, replace=False)
	batch_data = [ train_set[i]  for i in rand_indices] 	#[batch_size,2]  list, each item is a tuple
	x_input  = [ x[0]  for x in batch_data] 				#[batch_size,9]  list
	y_target = np.array([ y[1]  for y in batch_data]) 		#[batch_size,]   np.array
	sess.run(train_step, feed_dict={X: x_input, Y: y_target})

	temp_loss = sess.run(loss, feed_dict={X: x_input, Y: y_target})
	loss_vec.append(temp_loss)
	if j%500==0:
		print(j+1, " ", end="")
		print('iteration ' + str(j) + ' Loss: ' + str(temp_loss))


# Print loss
plt.plot(loss_vec, 'k-', label='Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# Make Prediction:
test_boards = [test_board]
feed_dict   = {X: test_boards}
logits      = sess.run(model_output, feed_dict=feed_dict) 	#[None,9]
predictions = sess.run(prediction,   feed_dict=feed_dict) 	#[None,]
print("Make Prediction: ")
print(predictions)

# Declare function to check for win
def check(board):
	wins = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
	for i in range(len(wins)):
		if board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == 1.:
			return(1)
		elif board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == -1.:
			return(1)
	return(0)


def my_check(board):
	wins = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
	for i in range(len(wins)):
		if board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == 1.:
			return(1)
		elif board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == -1.:
			return(-1)
	return(0)



'''
original board: 			 vertical (or horizontal) 垂直（或水平）
#   0  |  1  |  2
# -----------------
#   3  |  4  |  5
# -----------------
#   6  |  7  |  8

>>> wins = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,5,6]]
>>> np.array(wins)
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8],
       [0, 3, 6],
       [1, 4, 7],
       [2, 5, 8],
       [0, 4, 8],
       [2, 5, 6]])
>>> bo = moves[-1][0]
>>> bo
[1, 0, 0, -1, -1, 0, 1, 0, -1]
>>> i = 3
>>> bo[wins[i][0]]
1
>>> bo[wins[i][1]]
-1
>>> bo[wins[i][2]]
1
>>> wins[i]
[0, 3, 6]
>>> bo[wins[i][0]] == bo[wins[i][1]]
False
>>> bo[wins[i][0]] == bo[wins[i][1]] == bo[wins[i][2]]
False
>>> bo[wins[i][0]] == bo[wins[i][1]] == bo[wins[i][2]] == 1
False
>>> bo[wins[i][0]] == bo[wins[i][1]] == bo[wins[i][2]] == 1.
False
>>> 

check whether this board is ended?
'''


# Let's play against our model
game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
win_logical  = False
num_moves    = 0
while not win_logical:
	player_index = input('Input index of your move (0-8): ')
	num_moves += 1
	# Add player move to game
	game_tracker[int(player_index)] = 1.

	# Get model's move by first getting all the logits for each index
	[potential_moves] = sess.run(model_output, feed_dict={X: [game_tracker]})
	# Now find allowed moves (where game tracker values = 0.0)
	allowed_moves = [ ix  for ix,x in enumerate(game_tracker) if x==0.0]
	print("\tpotential_moves", "%s"%(' '*9), potential_moves)
	print("\tallowed_moves  ", "%s"%(' '*9), allowed_moves  )
	# Find best move by taking argmax of logits if they are in allowed moves
	model_move = np.argmax([ x  if ix in allowed_moves else -999.0  for ix,x in enumerate(potential_moves)])
	print("\tpotential_moves (updated)", [ x  if ix in allowed_moves else -999.0  for ix,x in enumerate(potential_moves)])
	print("\tmodel_move               ", model_move)

	# Add model move to game
	game_tracker[int(model_move)] = -1.
	print('Model has moved')
	print_board(game_tracker)
	# Now check for win or too many moves
	if my_check(game_tracker) == 1:
		print("Congratulations! You won~~") #win
	elif my_check(game_tracker) == -1:
		print("Sorry, you losed this game.")
	if check(game_tracker) == 1 or num_moves >=5:
		print('Game Over!')
		win_logical = True



'''
game_tracker 		#[9,]
potential_moves 	#[9,]
	model_output 		#[None,9] -> [1,9]
allowed_moves 		#[len1,]
					# len1 = 9 - number of '0.' in game_tracker
model_move 			#a number
	[x if ...]			#[9,]
						# change 'the disallowed position' into '-999.0'


>>> am = [0,1,3,4]
>>> pm = [0.,0., 1.,0.,0.,-1.,-1.,1.,1.]
>>> pm
[0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0]
>>> am
[0, 1, 3, 4]
>>> enumerate(pm)
<enumerate object at 0x7f2ba8041eb0>
>>> for ix,x in enumerate(pm):
...     print ix, x
... 
0 0.0
1 0.0
2 1.0
3 0.0
4 0.0
5 -1.0
6 -1.0
7 1.0
8 1.0
>>> for ix,x in enumerate(pm):
...     if ix in am:
...             print ix, x
...     else:
...             print ix, 'No'
... 
0 0.0
1 0.0
2 No
3 0.0
4 0.0
5 No
6 No
7 No
8 No
>>> 

'''




'''
exe-2th
ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ python ch6-8_nn_tic_tac_toe.py
Traceback (most recent call last):
  File "ch6-8_nn_tic_tac_toe.py", line 296, in <module>
    model_output = model(X, A1, A2, bias1, bias2)
  File "ch6-8_nn_tic_tac_toe.py", line 284, in model
    layer2 = tf.nn.add(tf.matmul(layer1, A2), bias2)
AttributeError: 'module' object has no attribute 'add'
ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ python ch6-8_nn_tic_tac_toe.py
2018-03-02 17:06:45.829559: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
iteration 0 Loss: 1.79747
iteration 0 Loss: 1.49647
iteration 0 Loss: 1.06577
iteration 0 Loss: 1.38808
iteration 0 Loss: 1.27466
iteration 0 Loss: 1.07873
iteration 0 Loss: 0.995785
iteration 0 Loss: 1.33242
iteration 0 Loss: 0.850847
iteration 0 Loss: 0.61486
iteration 0 Loss: 0.713415
iteration 0 Loss: 0.735654
iteration 0 Loss: 0.871088
iteration 0 Loss: 0.624634
iteration 0 Loss: 0.883735
iteration 0 Loss: 0.665212
iteration 0 Loss: 0.818838
Make Prediction: 
[6]
Input index of your move (0-8): 4
	potential_moves           [ 4.83380604  2.51915169  3.3205049   3.83043289  0.93352538  1.86657131
  5.51010799  1.86496258  3.93983459]
	allowed_moves             [0, 1, 2, 3, 5, 6, 7, 8]
	potential_moves (updated) [ 4.83380604  2.51915169  3.3205049   3.83043289  0.93352538  1.86657131
  5.51010799  1.86496258  3.93983459]
	model_move                6
Model has moved
   |   |  
-----------
   | X |  
-----------
 O |   |  
Input index of your move (0-8): 3
	potential_moves           [ 3.42774987  2.71229386  3.94125915  0.41175175  2.74738836 -1.63726234
  4.68063593  1.38495946  3.6067214 ]
	allowed_moves             [0, 1, 2, 5, 7, 8]
	potential_moves (updated) [ 3.42774987  2.71229386  3.94125915  0.41175175  2.74738836 -1.63726234
  4.68063593  1.38495946  3.6067214 ]
	model_move                2
Model has moved
   |   | O
-----------
 X | X |  
-----------
 O |   |  
Input index of your move (0-8): ^[[A1
Traceback (most recent call last):
  File "ch6-8_nn_tic_tac_toe.py", line 405, in <module>
    player_index = input('Input index of your move (0-8): ')
  Fi1e "<string>", line 1
    ^
SyntaxError: invalid syntax
ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ 




ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ python ch6-8_nn_tic_tac_toe.py
2018-03-02 17:11:18.680662: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
iteration 1 Loss: 2.02715
iteration 1 Loss: 1.20146
iteration 1 Loss: 1.29692
iteration 1 Loss: 1.32283
iteration 1 Loss: 1.22213
iteration 1 Loss: 1.06685
iteration 1 Loss: 1.12377
iteration 1 Loss: 0.981161
iteration 1 Loss: 0.861029
iteration 1 Loss: 1.19153
iteration 1 Loss: 0.911255
iteration 1 Loss: 0.747482
iteration 1 Loss: 0.620292
iteration 1 Loss: 0.623671
iteration 1 Loss: 0.792683
iteration 1 Loss: 0.749139
iteration 1 Loss: 0.717899
Make Prediction: 
[6]
Input index of your move (0-8): 4
	potential_moves           [ 0.94806623 -0.20688446  0.68356037 -3.08771229 -2.05117226 -0.17087018
 -0.81016827  0.43292254  1.26627207]
	allowed_moves             [0, 1, 2, 3, 5, 6, 7, 8]
	potential_moves (updated) [0.94806623, -0.20688446, 0.68356037, -3.0877123, -999.0, -0.17087018, -0.81016827, 0.43292254, 1.2662721]
	model_move                8
Model has moved
   |   |  
-----------
   | X |  
-----------
   |   | O
Input index of your move (0-8): 3
	potential_moves           [ 2.82757878  1.43656862 -0.0209074  -3.20837545 -1.49057913 -3.04396582
  1.35956001 -2.2601862   0.14509553]
	allowed_moves             [0, 1, 2, 5, 6, 7]
	potential_moves (updated) [2.8275788, 1.4365686, -0.020907402, -999.0, -999.0, -3.0439658, 1.35956, -2.2601862, -999.0]
	model_move                0
Model has moved
 O |   |  
-----------
 X | X |  
-----------
   |   | O
Input index of your move (0-8): 5
	potential_moves           [ 1.07389545  1.50007403  0.80666637 -6.51567316 -2.45702696 -1.30172443
 -1.27157235 -0.4180057   0.22069478]
	allowed_moves             [1, 2, 6, 7]
	potential_moves (updated) [-999.0, 1.500074, 0.80666637, -999.0, -999.0, -999.0, -1.2715724, -0.4180057, -999.0]
	model_move                1
Model has moved
 O | O |  
-----------
 X | X | X
-----------
   |   | O
Game Over!




exe-4th
ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ python ch6-8_nn_tic_tac_toe.py
2018-03-02 17:17:39.112077: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
0  iteration 1 Loss: 5.03448
0  iteration 1 Loss: 1.6496
0  iteration 1 Loss: 1.67443
0  iteration 1 Loss: 1.69587
0  iteration 1 Loss: 1.37155
0  iteration 1 Loss: 1.25894
0  iteration 1 Loss: 1.127
0  iteration 1 Loss: 1.16957
0  iteration 1 Loss: 0.831947
0  iteration 1 Loss: 1.02478
0  iteration 1 Loss: 1.04064
0  iteration 1 Loss: 1.04828
0  iteration 1 Loss: 0.970802
0  iteration 1 Loss: 1.10535
0  iteration 1 Loss: 0.856722
0  iteration 1 Loss: 0.955446
0  iteration 1 Loss: 0.823914
0  iteration 1 Loss: 0.866179
0  iteration 1 Loss: 0.977542
0  iteration 1 Loss: 0.752645
0  iteration 1 Loss: 0.596979
Make Prediction: 
[6]
Input index of your move (0-8): 4
	potential_moves           [ 4.83285904  1.15152836  1.61864567  0.63056326  2.17189336  2.50053525
  4.39917374  1.22736633  4.13802576]
	allowed_moves             [0, 1, 2, 3, 5, 6, 7, 8]
	potential_moves (updated) [4.832859, 1.1515284, 1.6186457, 0.63056326, -999.0, 2.5005352, 4.3991737, 1.2273663, 4.1380258]
	model_move                0
Model has moved
 O |   |  
-----------
   | X |  
-----------
   |   |  
Input index of your move (0-8): 3
	potential_moves           [ 1.39398253  0.75560665  1.70734358 -0.74197358  3.62164688  0.16611415
  4.5839963   2.85101104  4.71929741]
	allowed_moves             [1, 2, 5, 6, 7, 8]
	potential_moves (updated) [-999.0, 0.75560665, 1.7073436, -999.0, -999.0, 0.16611415, 4.5839963, 2.851011, 4.7192974]
	model_move                8
Model has moved
 O |   |  
-----------
 X | X |  
-----------
   |   | O
Input index of your move (0-8): 5
	potential_moves           [ 2.07692051  1.1822288   2.02039242 -3.15192986  2.27716017 -0.09536207
  3.25778675  2.89579034  5.18581438]
	allowed_moves             [1, 2, 6, 7]
	potential_moves (updated) [-999.0, 1.1822288, 2.0203924, -999.0, -999.0, -999.0, 3.2577868, 2.8957903, -999.0]
	model_move                6
Model has moved
 O |   |  
-----------
 X | X | X
-----------
 O |   | O
Congratulations! You won~~
Game Over!
ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ 





exe-5th
ubuntu@ubuntu-VirtualBox:~/Program/Code/tensorflow_cookbook$ python ch6-8_nn_tic_tac_toe.py
2018-03-02 17:19:42.471196: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
   1  iteration    0 Loss: 8.33294
 501  iteration  500 Loss: 1.71905
1001  iteration 1000 Loss: 1.59595
1501  iteration 1500 Loss: 1.31355
2001  iteration 2000 Loss: 1.36649
2501  iteration 2500 Loss: 1.28563
3001  iteration 3000 Loss: 1.09006
3501  iteration 3500 Loss: 1.00623
4001  iteration 4000 Loss: 1.16603
4501  iteration 4500 Loss: 0.851192
5001  iteration 5000 Loss: 1.04437
5501  iteration 5500 Loss: 0.963859
6001  iteration 6000 Loss: 0.932424
6501  iteration 6500 Loss: 0.965949
7001  iteration 7000 Loss: 0.670578
7501  iteration 7500 Loss: 0.904115
8001  iteration 8000 Loss: 0.926529
8501  iteration 8500 Loss: 0.755652
9001  iteration 9000 Loss: 0.883692
9501  iteration 9500 Loss: 0.824794
Make Prediction: 
[6]
Input index of your move (0-8): 2
	potential_moves           [ 4.91019297 -0.2876482   2.17918181  0.67864203  0.52589911  2.31134439
  0.68802369 -1.23942876  4.05522919]
	allowed_moves             [0, 1, 3, 4, 5, 6, 7, 8]
	potential_moves (updated) [4.910193, -0.2876482, -999.0, 0.67864203, 0.52589911, 2.3113444, 0.68802369, -1.2394288, 4.0552292]
	model_move                0
Model has moved
 O |   | X
-----------
   |   |  
-----------
   |   |  
Input index of your move (0-8): 3
	potential_moves           [ 3.58152843  0.50777829  3.36365557  0.74867058  0.60074091 -0.969576
  0.55225587 -2.31183696  4.55799389]
	allowed_moves             [1, 4, 5, 6, 7, 8]
	potential_moves (updated) [-999.0, 0.50777829, -999.0, -999.0, 0.60074091, -0.969576, 0.55225587, -2.311837, 4.5579939]
	model_move                8
Model has moved
 O |   | X
-----------
 X |   |  
-----------
   |   | O
Input index of your move (0-8): 7
	potential_moves           [ 3.98683667 -0.97029746  3.16416121  1.2696147  -0.87223136 -0.33330476
  2.7078352  -2.86805844  5.19429874]
	allowed_moves             [1, 4, 5, 6]
	potential_moves (updated) [-999.0, -0.97029746, -999.0, -999.0, -0.87223136, -0.33330476, 2.7078352, -999.0, -999.0]
	model_move                6
Model has moved
 O |   | X
-----------
 X |   |  
-----------
 O | X | O
Input index of your move (0-8): 5
	potential_moves           [ 3.45196533 -3.17330885  6.8414464  -2.57330918 -0.31977743  1.69315338
  3.09545183 -6.46786594  5.36859417]
	allowed_moves             [1, 4]
	potential_moves (updated) [-999.0, -3.1733088, -999.0, -999.0, -0.31977743, -999.0, -999.0, -999.0, -999.0]
	model_move                4
Model has moved
 O |   | X
-----------
 X | O | X
-----------
 O | X | O
Sorry, you losed this game.
Game Over!
ubuntu@ubuntu-VirtualBox:



'''

