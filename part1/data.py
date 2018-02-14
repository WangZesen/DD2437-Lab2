import math, random

def generate(kind=0, step=0.1, st=0, noise=0):
    x = []
    y = []
    if kind == 0:
        while st < math.pi * 2:
            x.append([st])
            y.append(math.sin(2 * st) + random.normalvariate(0, noise))
            st = st + step
    if kind == 1:
        while st < math.pi * 2:
            x.append([st])
            if math.sin(2 * st) + random.normalvariate(0, noise) > 0:
                y.append(1)
            else:
                y.append(-1)
            st = st + step
    return x, y
