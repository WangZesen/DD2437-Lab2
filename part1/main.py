import matplotlib.pyplot as plt
import numpy as np
from data import *
from net import *

def plot1D(X, y, label):
	x = np.array(X).T.tolist()[0]
	plt.plot(x, y, 'o', label = label)



# For 3.1

trainX, trainY = generate()
testX, testY = generate(st = 0.05)

n = 5
net = network(n)

net.leastSquares(trainX, trainY)

plot1D(testX, testY, "test")
plot1D(testX, net.forward(testX), "output")
plt.legend()
plt.show()

