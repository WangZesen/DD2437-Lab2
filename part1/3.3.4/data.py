import math, random

def generate(kind=0, step=0.1, st=0, noise=0, test = 0):
	x = []
	y = []
	if kind == 0:
		while st < math.pi * 2:
			x.append([st])
			y.append(math.sin(2 * st) + random.normalvariate(0, noise))
			st = st + step
		return x, y
	if kind == 1:
		while st < math.pi * 2:
			x.append([st])
			if math.sin(2 * st) + random.normalvariate(0, noise) > 0:
				y.append(1)
			else:
				y.append(-1)
			st = st + step
		return x, y
	if kind == 2:
		y1 = []
		file = None
		if test == 0:
			file = open('data/ballist.dat', 'r')
		else:
			file = open('data/balltest.dat', 'r')
		rawData = file.readlines()
		for i in range(len(rawData)):
			info = rawData[i].split('\t')
			x.append([float(info[0].split(' ')[0]), float(info[0].split(' ')[1])])
			y.append(float(info[1].split(' ')[0]))
			y1.append(float(info[1].split(' ')[1]))
		
		return x, y, y1
	
