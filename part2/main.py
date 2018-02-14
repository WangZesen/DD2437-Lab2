from data import *
from sys import argv
import numpy as np
import math
import matplotlib.pyplot as plt

assert len(argv) == 2

if argv[1] == '0': # Task 4.1
	n = 32
	m = 100
	maxIter = 20
	maxDist = 25
	node.dim = 84
	
	data = getData(0)
	nodeMap = topology(mode = 0, n = m)
	nodes = [node() for i in range(m)]

	for it in range(maxIter):
		for i in range(n):
			dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
			minNode = np.argmin(dists)
			neighbours = nodeMap.neighbour(minNode)
			curDist = round(maxDist - maxDist / (maxIter - 1) * it)
			for j in neighbours:
				nodes[j].update(data[i], curDist)

	names = getAuxilData(0)
	results = []
	for i in range(n):
		dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
		results.append(np.argmin(dists))
	args = np.argsort(np.array(results))
	for i in range(n):
		print (names[args[i]])

if argv[1] == "1":
	n = 10
	m = 10
	maxIter = 200
	maxDist = 2
	node.dim = 2
	
	data = getData(1)
	nodeMap = topology(mode = 1, n = m)
	nodes = [node() for i in range(m)]
	
	for it in range(maxIter):
		for i in range(n):
			dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
			minNode = np.argmin(dists)
			neighbours = nodeMap.neighbour(minNode)
			curDist = round(maxDist - maxDist / (maxIter - 1) * it)
			for j in neighbours:
				nodes[j].update(data[i], curDist)	
	
	results = []
	for i in range(n):
		dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
		results.append(np.argmin(dists))
	
	x = []
	y = []
	args = np.argsort(np.array(results))
	for i in range(n):
		x.append(data[args[i]][0])
		y.append(data[args[i]][1])
	plt.plot(x, y)
	plt.plot([x[0], x[-1]], [y[0], y[-1]])
	plt.plot(x, y, 'o')
	plt.show()

if argv[2] == "2":
	n = 349
	m = 10
	maxIter = 200
	maxDist = 4
	node.dim = 31
	
	data = getData(1)
	nodeMap = topology(mode = 1, n = m)
	nodes = [node() for i in range(m)]
	
	for it in range(maxIter):
		for i in range(n):
			dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
			minNode = np.argmin(dists)
			neighbours = nodeMap.neighbour(minNode)
			curDist = round(maxDist - maxDist / (maxIter - 1) * it)
			for j in neighbours:
				nodes[j].update(data[i], curDist)	
	
