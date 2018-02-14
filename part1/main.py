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
		trainX, trainY = generate(kind = kind, noise = 0.1)
		testX, testY = generate(kind = kind, st = 0.05)
		
		
		for sigma in range(1, 3):
			errors = []
			x = []
			for n in range(10, 20):
				net = network(n)
				for i in range(n):
					net.nodes[i].param[0] = [2 * math.pi / (n - 1) * i]
					net.nodes[i].param[1] = 2 ** (- sigma + 1)
				#net.leastSquares(trainX, trainY)
				net.deltaRule(trainX, trainY)
				x.append(n)
				errors.append(net.calError(trainX, trainY))

			plt.plot(x, errors, label = str(2 ** (- sigma + 1)))
		plt.legend()
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
		
	# For 3.3
	if argv[1] == '2':
		kind = 0
		n = 10
		net = network(n)
		trainX, trainY = generate(kind = kind)
		testX, testY = generate(kind = kind, st = 0.05)
		
		net.CL(trainX, trainY, deadNode = True)
		
		# Show the result of competition
		''' 
		x = []
		y = [0 for i in range(n)]
		for i in range(n):
			x.append(net.nodes[i].param[0][0])
			print (net.nodes[i].param[0])
		plot1D(trainX, trainY, 'train')
		plt.plot(x, y, 'o')
		plt.show()
		'''
		
		net.deltaRule(trainX, trainY)
		plot1D(testX, testY, 'test')
		plot1D(testX, net.forward(testX), 'network')
		plt.show()
		
		
