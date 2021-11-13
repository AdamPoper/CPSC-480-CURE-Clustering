# this is not CURE, just heirarchical clustering
# this is used to practice making a clustering algorithm

import numpy as np
import matplotlib.pyplot as plt
import math

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


class Entry:
    def __init__(self, key, val):
        self.key = key
        self.value = val

# i had to make my own map structure 
# because the map given by python wasn't what I was looking for
class ClusterMap:
    def __init__(self):
       self.entries = []
    def getValue(self, key):    # get the value at the key
        for e in self.entries:
            if e.key == key:
                return e.value
    def add(self, key, value):      # adds a new key value pair
        if not self.contains(key):
            self.entries.append(Entry(key, value))
    def contains(self, key):
        for e in self.entries:
            if e.key == key:
                return True
        return False
    def set(self, key, value):      # updates the value for a given key
        if not self.contains(key):
            self.add(key, value)
            return
        for e in self.entries:
            if e.key == key:
                e.value = value
                break
    

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
    # print(x)
    # print(y)
    
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

# returns a random set of vertices as an x array and a y array
def random_vertices(count):
    x = np.random.randint(low=1, high=100, size=count)
    y = np.random.randint(low=1, high=100, size=count)
    return x, y

# this will return vertices that are very obviously have their own clusters
# only used for debugging purposes
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
    x, y = random_vertices(vertex_count)
    # x, y = cluster_test_vertices(vertex_count)

    # static data for debugging purposes
    # x = [12, 9, 18, 70, 75, 85, 30, 60, 70, 20]
    # y = [60, 70, 66, 80, 77, 85, 50, 20, 22, 20]
   
    clusters = [] # the list of the clusters

    # make clusters out of the randomly generated x and y values
    for i in range(vertex_count):        
        v = Vertex(x[i], y[i])
        cluster = Cluster([v])
        clusters.append(cluster)  
    
    # num_clusters is the number of clusters that has been created by the algorithm 
    # it is first equal to the size of the array becasue each vertex is originally its own cluster
    num_clusters = len(clusters) 

    # this is the main algorithm and it will loop until the goal number of clusters has been reached
    # basically it works by finding the closest cluster for every cluster
    # then the algorithm puts all the closest pairs into map data structure
    # then it loops over the map and finds the pair of clusters that are closest to each other
    # it combines those clusters into a new cluster, deletes the old clusters and appends the new cluster
    # this is about the slowest way to do this but it actually works really well
    while num_clusters > goal_cluster_count:
        c_map = ClusterMap() # a cluster map will map a cluster to its closest neighbor
        # find all the closest points and add them to the map
        for i in range(len(clusters)):
            nearest_neighbor_distance = math.inf            
            for j in range(len(clusters)):
                if i != j:  # so a cluster doesn't consider itself as its closest neighbor
                    distance = euclidean_distance(clusters[i].centroid, clusters[j].centroid)
                    if distance < nearest_neighbor_distance:
                        nearest_neighbor_distance = distance
                        c_map.set(clusters[i], clusters[j])
        closest_pair = 0  # index of closest pair
        closest_pair_distance = math.inf  
        # loops over all the entries and finds the closest pair of clusters
        for i in range(len(c_map.entries)):
            e = c_map.entries[i]
            distance = euclidean_distance(e.key.centroid, e.value.centroid)
            if distance < closest_pair_distance:
                closest_pair = i
                closest_pair_distance = distance
        # make a new cluster, delete the old ones, and add the new one
        closest_clusters = c_map.entries[closest_pair]
        new_cluster = concat_clusters(closest_clusters.key, closest_clusters.value)
        clusters.remove(closest_clusters.key)
        clusters.remove(closest_clusters.value)
        clusters.append(new_cluster)
        num_clusters = len(clusters)
        # print(f'len clusters: {len(clusters)}')
        # print_clusters(clusters)
    draw_graph(clusters)

def main():
    count_vertices = 100
    desired_cluster_count = 30
    hierarchical_clustering(count_vertices, desired_cluster_count)

if __name__ == "__main__":
    main()