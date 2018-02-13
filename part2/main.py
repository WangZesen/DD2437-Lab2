from data import *
from sys import argv
import numpy as np
import math

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