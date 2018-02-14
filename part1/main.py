import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from data import *
from net import *
import math

def plot1D(X, y, label):
	x = np.array(X).T.tolist()[0]
	plt.plot(x, y, 'o', label = label)

if __name__ == "__main__":

	# For 3.1
	if argv[1] == '0':
		kind = 1
		trainX, trainY = generate(kind=kind)
		testX, testY = generate(kind=kind, st=0.05)

		# for sigma in range(3):
		for sigma in [1, 0.5, 0.2]:
			errors = []
			errors_t = []
			x = []
			N = len(trainX)
			for n in [5, 6, 7, 8, 9, 10, 11, 12]:
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
				print(net.calError(testX, testY))
				errors_t.append(net.calError(trainX, trainY))

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
		plot1D(testX, net.forward(testX), 'network')
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
		net.CL(trainX, trainY, deadNode = True)
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
				net.CL(trainX, trainY)
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
