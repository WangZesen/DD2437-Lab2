import matplotlib.pyplot as plt
from sys import argv
from data import *
import numpy as np
import math


assert len(argv) == 2

if argv[1] == '0': # Task 4.1
	n = 32
	m = 100
	maxIter = 100
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
			curDist = round(maxDist - maxDist / (maxIter - 1) * it)			
			neighbours = nodeMap.neighbour(minNode, curDist)
			for j in neighbours:
				nodes[j].update(data[i])	
	
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

if argv[1] == "2":
	n = 349
	m = 100
	maxIter = 20
	maxDist = 4
	node.dim = 31
	
	data = getData(2)
	nodeMap = topology(mode = 2, n = m)
	nodes = [node() for i in range(m)]
	
	for it in range(maxIter):
		for i in range(n):
			dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
			minNode = np.argmin(dists)
			curDist = round(maxDist - maxDist / (maxIter - 1) * it)			
			neighbours = nodeMap.neighbour(minNode, curDist)
			for j in neighbours:
				nodes[j].update(data[i])	
	
	results = []
	for i in range(n):
		dists = np.array([nodes[j].dist(data[i]) for j in range(m)])
		results.append(np.argmin(dists))
	
	args = np.argsort(np.array(results))
	district, party, sex = getAuxilData(2)

	# District
	print ("Result for district")
	votes = [[[0 for i in range(29)] for j in range(10)] for k in range(10)]
	
	for i in range(n):
		row = results[args[i]] // 10
		col = results[args[i]] - row * 10
		votes[row][col][district[i] - 1] += 1
	
	for i in range(10):
		for j in range(10):
			print (str(np.argmax(np.array(votes[i][j]))).ljust(3), end = " ")
		print ()
	
	# Party
	print ("Result for party")
	votes = [[[0 for i in range(7)] for j in range(10)] for k in range(10)]
	
	for i in range(n):
		row = results[args[i]] // 10
		col = results[args[i]] - row * 10
		votes[row][col][party[i] - 1] += 1
	
	for i in range(10):
		for j in range(10):
			print (str(np.argmax(np.array(votes[i][j]))).ljust(3), end = " ")
		print ()	

	# Sex
	print ("Result for sex")
	votes = [[[0 for i in range(2)] for j in range(10)] for k in range(10)]
	
	for i in range(n):
		row = results[args[i]] // 10
		col = results[args[i]] - row * 10
		votes[row][col][sex[i] - 1] += 1
	
	for i in range(10):
		for j in range(10):
			print (str(np.argmax(np.array(votes[i][j]))).ljust(3), end = " ")
		print ()	
