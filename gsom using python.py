#GSOM


from math import log, exp
import math
import random
import numpy as np
import scipy


class GSOM_Node:

    """ Represents one node in a growing SOM. """

    def __init__(
        self,
        dim,
        x,
        y
        ):
        """ Initialize this node. """

        print("creating a node")

        # Create a weight vector of the given dimension:
        # Initialize the weight vector with random values between 0 and 1.

        self.weights = np.random.rand(dim)

        # Remember the error occuring at this particular node

        self.error = 0.0

        # This node has no neighbours yet.

        self.right = None
        self.left = None
        self.up = None
        self.down = None

        # Copy the given coordinates.

        (self.x, self.y) = (x, y)

    def adjust_weights(self, target, learn_rate):
        """ Adjust the weights of this node. """

        for w in range(0, len(target)):
            self.weights[w] += learn_rate * (target[w] - self.weights[w])

    def is_boundary(self):
        """ Check if this node is at the boundary of the map. """

        if not self.right:
            return True
        if not self.left:
            return True
        if not self.up:
            return True
        if not self.down:
            return True
        return False


class GSOM:

    """ Represents a growing self-organizing map. """

    def _distance(self, v1, v2):
        """ Calculate the euclidean distance between two arrays."""

        dist = 0.0
        for (v, w) in zip(v1, v2):
            dist += pow(v - w, 2)
        return dist

    def _find_bmu(self, vec):
        """ Find the best matching unit within the map for the given input vector. """

        dist = float('inf')
        winner = False
        for node in self.nodes:
            d = self._distance(vec, node.weights)
            if d < dist:
                dist = d
                winner = node

        return winner

    def __init__(self, dataset, spread_factor=0.5):
        """ Initializes this GSOM using the given data. """

        
        # Determine the dimension of the data.

        self.dim = len(dataset[0])

        # Calculate the growing threshold:

        self._GT = -self.dim * math.log(spread_factor)

        # Create the 4 starting Nodes.

        self.nodes = []
        n00 = GSOM_Node(self.dim, 0, 0)
        n01 = GSOM_Node(self.dim, 0, 1)
        n10 = GSOM_Node(self.dim, 1, 0)
        n11 = GSOM_Node(self.dim, 1, 1)
        self.nodes.extend([n00, n01, n10, n11])

        # Create starting topology

        n00.right = n10
        n00.up = n01
        n01.right = n11
        n01.down = n00
        n10.up = n11
        n10.left = n00
        n11.left = n01
        n11.down = n10

        self.init_lr = 0.1  # Initial value of the learning rate
        self.alpha = 0.1

    def train(self):

        # Select the next input.
        print("trainig")

       

        

        

        # We now present the input several times to the network.

        for  input_ in dataset:

            for _ in range(100):

                

                    # Find the best matching unit
                    BMU = self._find_bmu(input_)
                                        # Calculate the learn rate.
                    # Note that the learning rate, according to the original paper,
                    # is resetted for every new input.

                    learn_rate = self.init_lr * self.alpha * (1 - 1.5/ len(self.nodes))

                    # Adapt the weights of the direct topological neighbours

                    neighbours = []
                    neighbours.append(BMU)
                    if BMU.left:
                        neighbours.append(BMU.left)
                    if BMU.right:
                        neighbours.append(BMU.right)
                    if BMU.up:
                        neighbours.append(BMU.up)
                    if BMU.down:
                        neighbours.append(BMU.down)

                    for node in neighbours:
                        node.adjust_weights(input_, learn_rate)

                    # Calculate the error.

                    err = self._distance(BMU.weights, input_)

                    # Add the error to the node.

                    nodes = self._node_add_error(BMU, err)



            


    def _node_add_error(self, node, error):
        """ Add the given error to the error value of the given node.
........    This will also take care of growing the map (if necessary) and
............distributing the error along the neighbours (if necessary) """

        node.error += error

        # Consider growing

        if node.error > self._GT:
            if not node.is_boundary():

                # Distribute the error along the neighbours.
                # Since this is not a boundary node, this node must have
                # 4 neighbours.

                node.error = 0.5 * self._GT
                node.left.error += 0.25 * node.left.error
                node.right.error += 0.25 * node.right.error
                node.up.error += 0.25 * node.up.error
                node.down.error += 0.25 * node.down.error

            nodes = self._grow(node)
            return nodes

        return 0

    def _grow(self, node):
        """ Grow this GSOM. """

        # We grow this GSOM at every possible direction.

        

        nodes = []
        if node.left == None:
            nn = self._insert(node.x - 1, node.y, node)
            nodes.append(nn)
            print("Growing left at: (" + str(node.x) + "," + str(node.y)\
					+ ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")


        if node.right == None:
            nn = self._insert(node.x + 1, node.y, node)
            nodes.append(nn)
            print("Growing right at: (" + str(node.x) + "," + str(node.y)\
					+ ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")


        if node.up == None:
            nn = self._insert(node.x, node.y + 1, node)
            nodes.append(nn)
            print("Growing up at: (" + str(node.x) + "," + str(node.y) +\
					") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

        if node.down == None:
            nn = self._insert(node.x, node.y - 1, node)
            nodes.append(nn)
            print("Growing down at: (" + str(node.x) + "," + str(node.y) +\
					") -> (" + str(nn.x) + ", " + str(nn.y) + ")")
        return nodes

    def _insert(
        self,
        x,
        y,
        init_node
        ):

        # Create new node

        new_node = GSOM_Node(self.dim, x, y)
        self.nodes.append(new_node)

        # Create the connections to possible neighbouring nodes.

        for node in self.nodes:

            # Left, Right, Up, Down

            if node.x == x - 1 and node.y == y:
                new_node.left = node
                node.right = new_node
            if node.x == x + 1 and node.y == y:
                new_node.right = node
                node.left = new_node
            if node.x == x and node.y == y + 1:
                new_node.up = node
                node.down = new_node
            if node.x == x and node.y == y - 1:
                new_node.down = node
                node.up = new_node

        # Calculate new weights, look for a neighbour.

        neigh = new_node.left
        if neigh == None:
            neigh = new_node.right
        if neigh == None:
            neigh = new_node.up
        if neigh == None:
            neigh = new_node.down
        if neigh == None:
            print( '_insert: No neighbour found!')

        for i in range(0, len(new_node.weights)):
            new_node.weights[i] = 2 * init_node.weights[i]  - neigh.weights[i]

        return new_node






dataset = np.genfromtxt('wine_dataset.txt',delimiter = ',')
#print(dataset)

num_of_datasample=len(dataset)
SF = 0.5
Test = GSOM(dataset, SF)
Test.train()
print(len(Test.nodes))
weight_matrix = [[None]*len(dataset[0])]*len(Test.nodes)



    
for s in range(len(Test.nodes)):
                    weight_matrix[s] = Test.nodes[s].weights
    
                    
                        
                    


print(weight_matrix)
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt





sse = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(weight_matrix)
    #dataset["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

