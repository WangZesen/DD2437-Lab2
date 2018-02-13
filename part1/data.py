import math, random

def generate(kind = 0, step = 0.1, st = 0, noise = 0):
	x = []
	y = []
	if kind == 0:
		while st < math.pi * 2:
			x.append([st])
			y.append(math.sin(st) + random.normalvariate(0, noise))
			st = st + step
	if kind == 1:
		while st < math.pi * 2:
			x.append([st])
			if st < math.pi:
				y.append(1 + random.normalvariate(0, noise))
			else:
				y.append(-1 +  + random.normalvariate(0, noise))
			st = st + step
	return x, y
