import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.style as style
import random

style.use('ggplot')

class K_Means:
    
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    
    # calc distance - Euclidean distance(for all dimentions)
    def Euclidean_distance(self, data_one, data_two):
 
        squared_dist = 0
        for i in range(len(data_one)):
            squared_dist += (data_one[i]-data_two[i])**2
        straight_distance = math.sqrt(squared_dist)

        return straight_distance
    

    # clustering
    # initialize the centroids, the first 'k' elements in the dataset will be initial centroids
    def fit(self, data):       
        self.centroids = {}
        randlist = random.sample(range(0, 150), self.k)
#         print(randlist)
        
        for i in range(self.k):
            self.centroids[i] = data[randlist[i]]

        # main loop
        for i in range(self.max_iterations):
            # need to cal distance for each iteration
            self.classes = {}
            # divided to k clusters for each iteration
            for j in range(self.k):
                self.classes[j] = []
            # find distance between point and each centroid; choose the nearest centroid   
            for point in data:
                distance = []
                for cen in self.centroids.values():
                    d = self.Euclidean_distance(point, cen)
                    distance.append(d)
                classification = distance.index(min(distance))
                self.classes[classification].append(point)

            # re-calc cluster centroids
            previous = dict(self.centroids)

            # average the cluster datapoints to re-calc the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            isOptimal = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr-original_centroid)/original_centroid*100.0)>self.tolerance:
                    isOptimal = False

                # break out of the main loop if the results are optimal
            if isOptimal:
                break
        
    # calculate score for final results
    def calculate_score(self, centroids, classes) -> float:
        score = 0.0
        for index in centroids:
            pointList = classes[index]
            sum_dist = 0.0
            for point in pointList:
                sum_dist += self.Euclidean_distance(point, centroids[index])
            score += sum_dist
        return score        


def main():
    
    df = pd.read_csv("clusters.txt", sep=",", header=None)
    df.columns = ["x", "y"]
    dataset = df.astype(float).values.tolist()
    X = df.values #returns a numpy array
    km = K_Means(k=4)
    finalCentroids = []
    finalClasses = {}
    finalScore = 100000

    for i in range(10):
        km.fit(X)
        score = km.calculate_score(km.centroids, km.classes)
        if score<finalScore:
            finalScore = score
            finalCentroids = km.centroids
            finalClasses = km.classes
        else:
            continue
    
    
    colors = 10*["r", "g", "c", "b", "k"]

    for centroid in finalCentroids:
        print(finalCentroids[centroid])
        plt.scatter(finalCentroids[centroid][0], finalCentroids[centroid][1], s=150, marker="x")
    
    for classification in finalClasses:
        color = colors[classification]
        for features in finalClasses[classification]:
            plt.scatter(features[0], features[1], color=color, s=30)
    
    for index in finalCentroids:
        label = "["+str(round(finalCentroids[index][0],2))+","+str(round(finalCentroids[index][1],2))+"]"
        x = finalCentroids[index][0]
        y = finalCentroids[index][1]

        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')

    plt.show()
  


if __name__=="__main__":
    main()
    
