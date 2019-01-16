import numpy as np
import math


class SOM:
    def __init__(self, x, y, dataset, learning_rate,threshold_1):
        
        
        #Initialize properties
        self._x = x
        self._y = y
        self._learning_rate = float(learning_rate)
        self._num_iter = 0
        self._threshold_1 = float(threshold_1)
        #self._threshold_2 = float(threshold_2)


        
        

        self.dim = len(dataset[0])


        #Initializing variables and placeholders
        self._weights = np.random.rand(x*y,self.dim)
        self._locations = _generate_index_matrix(x, y)

        
    def train(self):

        # Select the next input.
            print("trainig") 

        # We now present the input several times to the network.
            i=0
            for _ in range(100):
                for  input_ in dataset:
                
                    #Calculate learning rate and radius                                                                        
                    decay_function = np.subtract(1.0, np.divide(self._num_iter, 100*len(dataset)))
                    self._learning_rate = np.multiply(self._learning_rate, decay_function)
                    self._threshold_1 = np.multiply(self._threshold_1, decay_function)
 

                    self.input_matrix = [[None]*(self.dim)]*(self._x*self._y)

                    for s in range(self._x*self._y):
                        t=0
                    for t in range(self.dim):
                        self.input_matrix[s][t] = input_[t]

                    sub_result = np.subtract(self.input_matrix,self._weights)

                    squared_distances = np.power(sub_result,2)
                    distances = squared_distances.sum(1)
                
                    wining_neuron = np.argmin(distances)
                
                    min_indices = []

                    for i in range(self._x*self._y):
                        if distances[i]<self._threshold_1:
                            min_indices.append(i)
                
            
                    wining_co_ordinate = co_ordinate_of_neuron(wining_neuron,self._x,self._y)

                    #print(wining_co_ordinate)
                
                    for k in min_indices:
                        adjust_weights(self._weights[k], input_, self._learning_rate)

                    self._num_iter = self._num_iter + 1
                
                
            

def co_ordinate_of_neuron(index_of_distance_list,x,y):
    count = 0
    
    for i in range(x):
        for j in range(y):
            if index_of_distance_list==0:
                var=[0,0]
                return var
            count=count+1
            if count==index_of_distance_list:
                var=[i,j]
                return var



def _generate_index_matrix(x,y):
        return list(_iterator(x, y))


    
def _iterator(x, y):
        for i in range(x):
            for j in range(y):
                yield np.array([i, j])



def find_distance(list_1,list_2):
    val = [x1-x2 for (x1,x2) in zip(list_1,list_2)]
    val = np.power(val,2)
    return (math.sqrt(val[0]+val[1]))


def adjust_weights(weights, target, learn_rate):
        """ Adjust the weights of this node. """

        for w in range(0, len(target)):
            weights[w] += learn_rate * (target[w] - weights[w])


#main program


dataset = np.genfromtxt('wine_dataset.txt',delimiter = ',')
#print(dataset)

num_of_datasample=len(dataset)

Test = SOM(9,9,dataset,0.5,0.5)
Test.train()




import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



#print(Test.input_matrix)

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(Test._weights)
    #dataset["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()



    
