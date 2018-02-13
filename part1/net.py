import math, random
import numpy as np
from numpy.linalg import pinv

class node:
	dim = 1
	sigma = 1.0

	def __init__(self):
		self.param = [[random.uniform(-10, 10) for i in range(node.dim)], node.sigma]
	def dist(self, x):
		assert len(x) == node.dim
		count = 0
		for i in range(len(x)):
			count = count + (x[i] - self.param[0][i]) ** 2
		return count		
	def radial(self, x):
		assert len(x) == node.dim
		return math.e ** (- self.dist(x) / 2 / (self.param[1] ** 2))
	def update(self, x, eta = 0.1):
		assert len(x) == node.dim
		for i in range(node.dim):
			self.param[0][i] = self.param[0][i] + (x[i] - self.param[0][x]) * eta

class network:
	def __init__(self, n = 5):
		self.n = n
		self.nodes = [node() for i in range(n)]
		self.w = np.array([[random.uniform(0, 1) for i in range(n)]]).T
	def leastSquares(self, trainX, trainY):
		dataNum = len(trainX)
		phi = np.zeros((dataNum, self.n))
		for i in range(dataNum):
			for j in range(self.n):
				phi[i][j] = self.nodes[j].radial(trainX[i])		
		npTrainY = np.array([trainY]).T
		self.w = np.dot(np.dot(np.dot(pinv(phi), pinv(phi.T)), phi.T), npTrainY)
		assert self.w.shape == (self.n, 1)
	def deltaRule(self, trainX, trainY, lr = 0.1, maxIter = 3000, batch = 1):
		assert batch <= len(trainX)
		for k in range(maxIter):
			samples = random.sample(range(len(trainX)), batch)
			delta = np.zeros((self.n, 1))
			for it in range(batch):
				index = samples[it]
				vectorPhi = np.zeros((self.n, 1))
				for i in range(self.n):
					vectorPhi[i][0] = self.nodes[i].radial(trainX[index])
				e = trainY[index] - np.dot(vectorPhi.T, self.w)
				delta = delta + lr * e * vectorPhi
			self.w = self.w + delta / batch
			
			
	def CLDeltaRule(self, trainX, trainY, lr = 0.1, maxIter = 3000, deadNode = False):
		# batch size is fixed to 1
		if deadNode == False:
			for k in range(maxIter):
				samples = random.sample(range(len(trainX)), batch)
				index = random.randint(0, len(trainX) - 1)
				vectorPhi = np.zeros((self.n, 1))
				minDist = 1e+20
				minNode = -1
				for i in range(self.n):
					if self.nodes[i].dist(trainX[index]) < minDist:
						minDist = self.nodes[i].dist(trainX[index])
						minNode = i
				self.nodes[minNode].update(trainX[index])
				e = trainY[index] - np.dot(vectorPhi.T, self.w)
				delta = delta + lr * e * vectorPhi
				self.w = self.w + delta / batch
			
			
	def calError(self, testX, testY):
		results = self.forward(testX)
		count = 0
		for i in range(len(results)):
			count = count + abs(results[i] - testY[i])
		return count / len(results)
	def forward(self, testX):
		dataNum = len(testX)
		phi = np.zeros((dataNum, self.n))
		for i in range(dataNum):
			for j in range(self.n):
				phi[i][j] = self.nodes[j].radial(testX[i])
		return np.dot(phi, self.w).T.tolist()[0]
	def squareForward(self, testX):
		dataNum = len(testX)
		phi = np.zeros((dataNum, self.n))
		for i in range(dataNum):
			for j in range(self.n):
				phi[i][j] = self.nodes[j].radial(testX[i])
		results = np.dot(phi, self.w).T.tolist()[0]		
		for i in range(dataNum):
			if results[i] > 0:
				results[i] = 1
			else:
				results[i] = -1
		return results
