#!/usr/bin/python
# -*- coding: utf-8 -*-

# This implements the FastMap algorithm for mapping points
# where only the distance between them is known to N-
# dimensional coordinates.
# 
# The FastMap algorithm was published in:
#
# Christos Faloutsos and King-Ip (David) Lin, 1995. FastMap:
# a fast algorithm for indexing, data-mining and
# visualization of traditional and multimedia datasets. 
# Proceedings of the ACM SIGMOD International Conference on
# Management of Data, Michael J. Carey and Donovan A. Schneider,
# eds., San Jose, California, 163-174.
#
# author: XU Shuo (pzczxs@gmail.com)
# modified by 
# Chenqi Liu 2082-6026-02 
# Che-Pai Kung 5999-9612-95 
# Mengyu Zhang 3364-2309-80
# 

import math
import random
import scipy
import sys
import matplotlib.pyplot as plt


# we will repeat the pick-pivot points heuristic this many times
# a higher value means "better" results, but 1 also works well
DISTANCE_ITERATIONS = 5

class FastMap: 

    def __init__(self, dist, verbose=False):
        """dist is a NxN distance matrix"""
        self.dist = dist
        self.verbose = verbose

    def _furthest(self, o): 
        mx = -scipy.inf
        idx = -1
        for i in range(len(self.dist)): 
            d = self._dist(i, o, self.col)
            if d > mx: 
                mx = d
                idx = i

        return idx

    def _pickPivot(self):
        """Find the two most distant points"""
        random.seed(0)
        o1 = random.randint(0, len(self.dist)-1)
        o2 = -1

        for i in range(DISTANCE_ITERATIONS):
            o = self._furthest(o1)
            if o == o2:
                break
            o2 = o
            
            o = self._furthest(o2)
            if o == o1:
                break
            o1 = o

        self.pivots[self.col] = (o1, o2)
        return (o1, o2)


    def _map(self, K): 
        if K == 0:
            return 
    
        px, py = self._pickPivot()

        if self.verbose:
            print ("Picked %d, %d at K = %d" %(px, py, K))

        if self._dist(px, py, self.col) == 0: 
            return
        
        for i in range(len(self.dist)):
            self.coords[i][self.col] = self._x(i, px, py)

        self.col += 1
        self._map(K - 1)

    def _x(self, i, x, y, outofsample=False):
        """Project the i'th point onto the line defined by x and y"""
        dix = self._dist(i, x, (outofsample and [self.col2]
                                or [self.col])[0], outofsample)
        diy = self._dist(i, y, (outofsample and [self.col2]
                                or [self.col])[0], outofsample)
        dxy = self._dist(x, y, (outofsample and [self.col2]
                                or [self.col])[0])
        
        return (dix + dxy - diy) / (2*math.sqrt(dxy))

    def _dist(self, x, y, k, outofsample=False): 
        """Recursively compute the distance based on previous projections"""
        if k == 0:
            return (outofsample and [self.dist2[x, y]**2]
                    or [self.dist[x, y]**2])[0]

        rec = self._dist(x, y, k-1, outofsample)
        if outofsample:
            resd = (self.coords2[x][k-1] - self.coords[y][k-1])**2
        else:
            resd = (self.coords[x][k-1] - self.coords[y][k-1])**2
        
        return (rec - resd)


    def _distimage(self):
        dist = scipy.zeros((len(self.dist), len(self.dist)), scipy.double)

        for x in range(len(self.dist) - 1):
            for y in range(x + 1, len(self.dist)):
                # the Eucliean distance
                dist[x, y] = math.sqrt(sum((self.coords[x] - self.coords[y])**2))
                dist[y, x] = dist[x, y]

        return dist

    def map(self, K):
        """returns coordinates for each N in K dimensions"""
        self.col = 0
        self.coords = scipy.zeros((len(self.dist), K), scipy.double)
        self.pivots = scipy.zeros((K, 2), scipy.int64)
        
        self._map(K)
        
        return self.coords

    def stress(self):
        """the stree function"""
        dist_image = self._distimage()
        
        ret = 0.0
        factor = 0.0
        for x in range(len(self.dist)):
            for y in range(len(self.dist)):
                ret += (self.dist[x, y] - dist_image[x, y])**2
                factor += self.dist[x, y]**2

        return math.sqrt(ret / factor)

    def getPivots(self):
        """return the pivots"""
        return self.pivots

    def outOfSample(self, dist):
        self.dist2 = dist
        self.coords2 = scipy.zeros((len(dist), self.col), scipy.double)

        for i in range(len(self.dist2)):
            self.col2 = 0
            for (p1, p2) in self.pivots:
                self.coords2[i][self.col2] = self._x(i, p1, p2, True)
                self.col2 += 1

        return self.coords2
    
if __name__ == "__main__":
    
    data =[]
    X = []
    with open('fastmap-data.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split('	')]
            data.append(inner_list)

    for i in range(10):
        row = []
        for j in range(10):
            row.append(0)
        X.append(row)
    
    for p in data:
        i = int(p[0]) - 1
        j = int(p[1]) - 1 
        X[i][j]= int(p[2])
        X[j][i]= int(p[2])

    # print(X)
    dist = scipy.array(X)
    # dist2 = scipy.array([[0, 1, 1, 100, 100],
    #                      [2, 2, 2, 50, 50]])
    # print(dist2)
    fm = FastMap(dist, True)
     
    # print fm.map(2)

    # print fm.getPivots()
    result = fm.map(2)

    labels = []
    with open('fastmap-wordlist.txt') as f:
        for line in f:
            labels.append(line[:])
    # print(labels)
    #Draw results
    for i,label in enumerate(labels):
        x = result[i][0]
        y = result[i][1]
        plt.scatter(x, y, s=30)
        plt.text(x+0.1, y+0.1, label, fontsize=9)

    plt.show()



    # print fm.outOfSample(dist2)

##    usage = """Usage: fastmap.py -dimension K distFile outFile"""
##    argv = sys.argv
##    if len(argv) < 3:
##        print usage
##        sys.exit(1)
##
##    # process command line options
##    distFile = argv[-2]
##    outFile = argv[-1]
##    K = 2
##
##    i = 1
##    while i < len(argv) - 2:
##        if argv[i] == "-dimension":
##            i += 1
##            K = int(argv[i])
##        else:
##            print usage
##            sys.exit(1)
##        i += 1
##
##    dist = scipy.loadtxt(distFile, dtype = scipy.double, delimiter = " ")
##
##    scipy.savetxt(outFile, FastMap(dist, True).map(K), fmt = "%g", delimiter = " ")
    
    
