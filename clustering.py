#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from inspect import currentframe

# a vertex is just an x position and a y position
# this makes is easier for grouping vertices together 
# rather than having 2 arrays for the x and y positions I can have 1 array with the x and y positions together
# I could have just used tuples or something but I prefer to have my own class
class Vertex:
    def __init__(self, _x=0, _y=0):
        self.x = _x
        self.y = _y
    def __repr__(self):
        return f"({self.x}, {self.y})"

# A Cluster is an array of vertices, a centroid, and a radius
# this makes it easier than having a multi-dimensional array of vertices
# each cluster can easily calculate its own centroid
class Cluster:
    def __init__(self, _vertices=[]):
        self.vertices = _vertices
        self.centroid = Vertex()
        self.calc_centroid()
        self.radius = self.calc_radius()
    def __repr__(self):
        return str(self.vertices)
    def calc_centroid(self):
        x_sum = 0
        y_sum = 0
        for c in self.vertices:
            x_sum += c.x
            y_sum += c.y
        self.centroid = Vertex(x_sum/len(self.vertices), y_sum/len(self.vertices))   
    def calc_radius(self):
        radius = 0
        for v in self.vertices:            
            distance = euclidean_distance(v, self.centroid)
            if distance > radius:
                radius = distance
        return radius        

# calculates the number of vertices in a list of clusters
def num_vertices(clusters):
    count = 0
    for c in clusters:
        count += len(c.vertices)
    return count

# concatenates 2 clusters and returns the new cluster
def concat_clusters(c1, c2):
    vertices = c1.vertices + c2.vertices
    return Cluster(vertices)

# calculates the euclidean distance between 2 vertices
def euclidean_distance(v1, v2):
    dx = abs(v2.x-v1.x)
    dy = abs(v2.y-v1.y)    
    return math.sqrt(dx**2 + dy**2)

# draws a graph with x and y values
def draw_graph(clusters):
    x, y = get_raw_xy(clusters)
    print(x)
    print(y)
    
    fig, ax = plt.subplots()
    
    for cluster in clusters:
        _x = cluster.centroid.x
        _y = cluster.centroid.y
        circle = plt.Circle((_x, _y), cluster.radius+1, color='blue', fill=False)
        ax.add_patch(circle)

    ax.set_ylim(1,100)
    ax.set_xlim(1,100)

    ax.plot(x, y, 'o', color='black')

    plt.show()

# gets the raw x and y data as their own lists from a list of clusters
def get_raw_xy(clusters):
    x = []
    y = []
    for c in clusters:
        for v in c.vertices:
            x.append(v.x)
            y.append(v.y)
    return x, y

# not currently used by the algorithm
def f(avg, density):
    a = 0.2
    return a - ((-avg/density)+1)*a

# not currently used by the algorithm
def calc_density(cluster):
    max_x, max_y = 0
    min_x, min_y = math.inf
    for vertex in cluster:
        if vertex.x > max_x:
            max_x = vertex.x
        if vertex.y > max_y:
            max_y = vertex.y
        if vertex.x < min_x:
            min_x = vertex.x
        if vertex.y < min_y:
            min_y = vertex.y
    x = max_x - min_x
    y = max_y - min_y
    area = x * y
    density = len(cluster.vertices) / area
    return density

# print information about the clusters
def print_clusters(clusters):
    print(f'Cluster Count: {len(clusters)}')
    print(f'Vertex Count: {num_vertices(clusters)}')
    for c in clusters:
        for v in c.vertices:
            print(v, end='')
        print('\n')

def random_vertices(count):
    x = np.random.randint(low=1, high=100, size=count)
    y = np.random.randint(low=1, high=100, size=count)
    return x, y

def cluster_test_vertices(count):
    vertices = int(count/4)
    lowest_low = 5
    highest_low = 25
    lowest_high = 75
    highest_high = 95

    x0 = np.random.randint(low=lowest_low, high=highest_low, size=vertices)
    y0 = np.random.randint(low=lowest_low, high=highest_low, size=vertices)

    x1 = np.random.randint(low=lowest_low, high=highest_low,   size=vertices)
    y1 = np.random.randint(low=lowest_high, high=highest_high, size=vertices)

    x2 = np.random.randint(low=lowest_high, high=highest_high, size=vertices)
    y2 = np.random.randint(low=lowest_high, high=highest_high, size=vertices)

    x3 = np.random.randint(low=lowest_high, high=highest_high, size=vertices)
    y3 = np.random.randint(low=lowest_low, high=highest_low,  size=vertices)

    x = np.concatenate((x0, x1, x2, x3), axis=0)
    y = np.concatenate((y0, y1, y2, y3), axis=0)
    return x, y
    

# this is the main heirarchical clustering algorithm
def hierarchical_clustering(vertex_count, goal_cluster_count):
    # generates random x and y values for the vertices
    # x, y = random_vertices(vertex_count)
    x, y = cluster_test_vertices(vertex_count)

    # static data for debugging purposes
    # x = [12, 9, 18, 70, 75, 85, 30, 60, 70, 20]
    # y = [60, 70, 66, 80, 77, 85, 50, 20, 22, 20]
   
    clusters = [] # the list of the clusters

    # make vertices out of the randomly generated x and y values
    for i in range(vertex_count):        
        v = Vertex(x[i], y[i])
        cluster = Cluster([v])
        clusters.append(cluster)  
    
    # num_clusters is the number of clusters that has been created by the algorithm 
    # it is first equal to the size of the array becasue each vertex is originally its own cluster
    num_clusters = len(clusters) 

    # this is the main algorithm and it will loop until the goal number of clusters has been reached
    # basically it works by checking each cluster against each other cluster and finding it's nearest neighbor
    # and it keeps doing this until the goal number of clusters has been reached
    # the time complexity is probably something like O(n*log n*num_clusters)
    # it's about the slowest and most brute force way to do this, but at least it works
    while num_clusters > goal_cluster_count:
        for i in range(len(clusters)):            
            nearest_neighbor = clusters[i+1]
            nearest_neighbor_distance = euclidean_distance(clusters[i].centroid, nearest_neighbor.centroid)
            print(f'Iteration: {i}')
            for j in range(len(clusters)):
                if i != j: 
                    other_distance = euclidean_distance(clusters[j].centroid, clusters[i].centroid)
                    # checks if this distance is less than the current nearest distance        
                    if other_distance <= nearest_neighbor_distance:
                        nearest_neighbor = clusters[j]
                        nearest_neighbor_distance = other_distance    
            print(f'The nearest neigbor to {clusters[i].centroid} is {nearest_neighbor.centroid}')   
            # concatenate the clusters into one new cluster    
            new_cluster = concat_clusters(clusters[i], nearest_neighbor)     
            # remove the 2 old clusters and just add in the new cluster       
            clusters.remove(clusters[i])
            clusters.remove(nearest_neighbor)       
            clusters.append(new_cluster)
            #draw_graph(clusters)
            #print_clusters(clusters)    
            #print('----------------------------------------------------------')       
            if i <= len(clusters):
                break   
        num_clusters = len(clusters)
    #print_clusters(clusters)
    draw_graph(clusters)

def main():
    count_vertices = 20
    desired_cluster_count = 4
    hierarchical_clustering(count_vertices, desired_cluster_count)

if __name__ == "__main__":
    main()