import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from data import *
from net import *
import math, copy

def plot1D(X, y, label):
	x = np.array(X).T.tolist()[0]
	plt.plot(x, y, label = label)

def cal2DError(net, net1, testX, testY, testY1):
	results = net.forward(testX)
	results1 = net1.forward(testX)
	
	count = 0
	for i in range(len(testX)):
		count = count + ((results[i] - testY[i]) ** 2 + (results1[i] - testY1[i]) ** 2) ** 0.5
		
	return count

def plotPoints1(X):
	x = []
	y = []
	for i in range(len(X)):
		x.append(X[i][0])
		y.append(X[i][1])
	plt.plot(x, y, 'o', alpha = 0.8, markersize = 20)

def plotPoints(X):
	x = []
	y = []
	for i in range(len(X)):
		x.append(X[i][0])
		y.append(X[i][1])
	plt.plot(x, y, 'o')

if __name__ == "__main__":

	# For 3.1
	if argv[1] == '0':
		kind = 0
		trainX, trainY = generate(kind=kind, noise = 0)
		testX, testY = generate(kind=kind, st=0.05)

		# for sigma in range(3):
		for sigma in [10, 1.5, 1, 0.5, 0.2]:
			errors = []
			errors_t = []
			x = []
			N = len(trainX)
			for n in range(5, 50):
			# for n in [N]:
				net = network(n)
				for i in range(n):
					net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
					# net.nodes[i].param[0] = trainX[i]
					# net.nodes[i].param[1] = 2 ** (- sigma)
					net.nodes[i].param[1] = sigma
				net.leastSquares(trainX, trainY)
				# net.deltaRule(trainX, trainY, batch=1, maxIter=5000, lr=0.05)
				x.append(n)
				errors.append(net.calError(testX, testY))
				print(net.calError(trainX, trainY))
				errors_t.append(net.calError(testX, testY))

			plt.plot(x, errors, label=sigma)
			# plt.plot(x, errors_t, 'k:', label=sigma)
		plt.legend()
		plt.xlabel("Number of units")
		plt.ylabel("Error")
		# plt.ylim((0, 0.2))
		plt.yscale('log')
		plt.show()

		'''
		# For visualize one case
		n = 7
		net = network(n)

		for i in range(n):
			net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
			#net.nodes[i].param[1] = 0.5

		# net.deltaRule(trainX, trainY)
		net.leastSquares(trainX, trainY)
		plot1D(testX, testY, 'data')
		plot1D(testX, net.squareForward(testX), 'transform', marker='.')
		plot1D(testX, net.forward(testX), 'output', marker='g')
		plt.legend()
		plt.show()
		'''
		
	# For 3.2
	if argv[1] == '1':
		kind = 0
		trainX, trainY = generate(kind=kind, noise=0.1)
		# testX, testY = generate(kind=kind, st=0.05)

		# for sigma in [1, 0.5, 0.2]:
		# 	errors = []
		# 	x = []
		#
		# 	for n in [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
		# 		net = network(n)
		# 		for i in range(n):
		# 			net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
		# 			net.nodes[i].param[1] = sigma
		# 		# net.leastSquares(trainX, trainY)
		# 		net.deltaRule(trainX, trainY, batch=1, maxIter=5000, lr=0.1)
		# 		x.append(n)
		# 		errors.append(net.calError(trainX, trainY))
		#
		#
		# 	# random distribution of RBF positioning
		# 	'''
		# 	for n in [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
		# 		errors_sum = 0
		# 		for j in range(5):
		# 			net = network(n)
		# 			for i in range(n):
		# 				# net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
		# 				net.nodes[i].param[0] = [random.uniform(0, 2 * math.pi)]
		# 				net.nodes[i].param[1] = sigma
		# 			net.leastSquares(trainX, trainY)
		# 			# net.deltaRule(trainX, trainY, batch=1, maxIter=5000, lr=0.05)
		# 			errors_sum += net.calError(trainX, trainY)
		# 		x.append(n)
		# 		errors.append(errors_sum)
		# 	'''

		# change eta (learning rate)
		for lr in [0.3, 0.1, 0.05, 0.01]:
			sigma = 0.5
			n = 10
			errors = []
			x = []

			for iters in range(0, 3001, 50):
				net = network(n)
				for i in range(n):
					net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
					net.nodes[i].param[1] = sigma
				# net.leastSquares(trainX, trainY)
				net.deltaRule(trainX, trainY, batch=1, maxIter=iters, lr=lr)
				x.append(iters)
				errors.append(net.calError(trainX, trainY))

			plt.plot(x, errors, label=lr)
		plt.legend()
		plt.xlabel("Number of iters")
		plt.ylabel("Error")
		# plt.yscale('log')
		plt.show()

		# For visualize one case
		'''
		n = 150
		net = network(n)

		for i in range(n):
			net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
			#net.nodes[i].param[1] = 0.5

		net.deltaRule(trainX, trainY, lr = 0.001, maxIter = 20000)

		plot1D(testX, testY, 'test')
		#plot1D(testX, net.squareForward(testX), 'normalized')
		plot1D(testX, net.r, it is the modulo operation.

forward(testX), 'network')
		print ("absolute residual error =", net.calError(testX, testY))
		plt.legend()
		plt.show()
		'''
		
	if argv[1] == '2': # For 3.3.1 Competitive Learning
		kind = 0
		trainX, trainY = generate(kind=kind, noise = 0.0)
		testX, testY = generate(kind=kind, st=0.05)
		
		
		
		testN = 10
		net = network(testN)
		net.CL(trainX, trainY, deadNode = False)
		for i in range(testN):
			plt.plot([net.nodes[i].param[0][0]], [0], 'ro')
		plot1D(trainX, trainY, 'train')
		plt.show()
		'''
		
		# for sigma in range(3):
		for sigma in [1, 0.5, 0.2]:
			errors = []
			errors_t = []
			x = []
			N = len(trainX)
			for n in range(5, 13):
			# for n in [N]:
				net = network(n)
				for i in range(n):
					net.nodes[i].param[1] = sigma
				net.CL(trainX, trainY, deadNode = True)
				net.leastSquares(trainX, trainY)
				# net.deltaRule(trainX, trainY, batch=1, maxIter=5000, lr=0.05)
				x.append(n)
				errors.append(net.calError(testX, testY))
				print(net.calError(trainX, trainY))
				errors_t.append(net.calError(testX, testY))

			plt.plot(x, errors_t, label=sigma)
			# plt.plot(x, errors_t, 'k:', label=sigma)
		plt.legend()
		plt.xlabel("Number of units")
		plt.ylabel("Error")
		# plt.ylim((0, 0.2))
		plt.yscale('log')
		plt.show()
		'''
	if argv[1] == '3':
		kind = 2
		trainX, trainY, trainY1 = generate(kind = kind)
		testX, testY, testY1 = generate(kind = kind, test = 1)
		node.dim = 2
		
		plotPoints(trainX)

		
		testN = 60
		net = network(testN)
		net.CL(trainX, trainY, deadNode = True)
		netNodes = []
		for i in range(testN):
			netNodes.append(copy.copy(net.nodes[i].param[0]))
		plotPoints1(netNodes)
		plt.show()
		
		
		# for sigma in range(3):
		for sigma in [1.0, 0.5, 0.2]:
			errors = []
			errors_t = []
			x = []
			N = len(trainX)
			for n in range(50, 100):
			# for n in [N]:
				net = network(n)
				for i in range(n):
					net.nodes[i].param[1] = sigma
				net.CL(trainX, trainY, deadNode = True)
				net1 = network(n)
				for i in range(n):
					net1.nodes[i].param[0] = copy.copy(net.nodes[i].param[0])
					net1.nodes[i].param[1] = sigma				

				net.leastSquares(trainX, trainY)
				net1.leastSquares(trainX, trainY1)				
				# net.deltaRule(trainX, trainY, batch=1, maxIter=5000, lr=0.05)
				x.append(n)
				errors.append(cal2DError(net, net1, trainX, trainY, trainY1))
				# print(net.calError(trainX, trainY))
				errors_t.append(cal2DError(net, net1, testX, testY, testY1))

			plt.plot(x, errors, label= "train" + str(sigma))
			plt.plot(x, errors_t, '--', label= "test" + str(sigma))			
			# plt.plot(x, errors_t, 'k:', label=sigma)
		plt.legend()
		plt.xlabel("Number of units")
		plt.ylabel("Error")
		# plt.ylim((0, 0.2))
		plt.yscale('log')
		plt.show()		
		
		
