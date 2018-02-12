import matplotlib.pyplot as plt
import numpy as np
from data import *
from net import *
import math

def plot1D(X, y, label):
	x = np.array(X).T.tolist()[0]
	plt.plot(x, y, 'o', label = label)



# For 3.1

kind = 0
trainX, trainY = generate(kind = kind)
testX, testY = generate(kind = kind, st = 0.05)

errors = []
x = []
for sigma in range(3):
	for n in range(5, 12):
		net = network(n)
		for i in range(n):
			net.nodes[i].param[0] = [math.pi / (n - 1) * i]
			net.nodes[i].param[1] = 5 ** (- sigma + 1)
		net.leastSquares(trainX, trainY)
		x.append(n)
		errors.append(net.calError(testX, testY))

	plt.plot(x, errors, label = str(sigma))
plt.show()
