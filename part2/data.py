import math, random
import numpy as np

class node:
	dim = 2
	def __init__(self):
		self.pos = [random.uniform(0, 1) for i in range(node.dim)]
	def dist(self, x):
		assert node.dim == len(x)
		count = 0
		for i in range(node.dim):
			count = count + (self.pos[i] - x[i]) ** 2
		return count
	def update(self, x, eta = 0.2):
		assert node.dim == len(x)
		for i in range(node.dim):
			self.pos[i] = self.pos[i] + (x[i] - self.pos[i]) * eta

class topology:
	def __init__(self, mode = 0, n = 5):
		self.mode = mode
		self.n = n
		if mode == 2:
			self.a = int(math.sqrt(self.n))
	def neighbour(self, x, dist = 1):
		neighbours = []
		if self.mode == 0: # Linear
			for i in range(self.n):
				if abs(i - x) <= dist:
					neighbours.append(i)
			return neighbours
		elif self.mode == 1: # Cycle
			for i in range(self.n):
				if abs(i - x) <= dist or abs(i - x - self.n) <= dist or abs(i - x + self.n) <= dist:
					neighbours.append(i)
			return neighbours
		elif self.mode == 2: # Grid
			r = x // self.a
			c = x - r * self.a
			for i in range(self.a):
				for j in range(self.b):
					if abs(i - r) + abs(c - j) <= dist:
						neighbours.append(i * self.a + j)
			return neighbours
			
def getAuxilData(dataSet = 0):
	if dataSet == 0:
		file = open('data/animalnames.txt', 'r')
		rawData = file.readlines()
		data = []
		for i in range(32):
			data.append(rawData[i].split('\'')[1])
		return data


def getData(dataSet = 0):
	if dataSet == 0:
		file = open('data/animals.dat', 'r')
		rawData = file.readlines()[0].split(',')
		data = [[0 for col in range(84)] for row in range(32)]
		for i in range(32):
			for j in range(84):
				data[i][j] = int(rawData[i * 84 + j])
		return data
	if dataSet == 1:
		file = open('data/cities.dat', 'r')
		rawData = file.readlines()
		data = [[0 for col in range(2)] for row in range(10)]
		for i in range(10):
			data[i][0] = float(rawData[i].split(',')[0])
			data[i][1] = float(rawData[i].split(',')[1][0:7])
		return data
	if dataSet == 2:
		file = open('data/votes.dat', 'r')
		rawData = file.readlines()[0].split(',')
		data = [[0 for col in range(31)] for row in range(349)]
		for i in range(349):
			for j in range(31):
				data[i][j] = float(rawData[i * 31 + j])
		return data

if __name__ == "__main__":
	print (getAuxilData(0))