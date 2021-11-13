# this is CURE clustering without any of our modifications
# this exists just to try to get it to work correctly

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# a vertex in this algorithm contains the data for a person
# the x value is population, the y value is gdp, 
class Vertex:
    def __init__(self, _x=0, _y=0):
        self.x = _x
        self.y = _y                
    def __repr__(self):
        return f"(x: {self.x}, y: {self.y})"

# A Cluster is an array of vertices, a centroid, and a radius
# this makes it easier than having a multi-dimensional array of vertices
# each cluster can easily calculate its own centroid
class Cluster:
    def __init__(self, _vertices=[]):
        self.vertices = _vertices
        self.centroid = Vertex()
        self.calc_centroid()
        self.radius = self.calc_radius()
        self.representative_points = [] 
        self.best_scattered_points = []
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
    def calc_representative_points(self):
        # if the cluster only has one point
        # the reps are just the one point
        if len(self.vertices) == 1:
            self.best_scattered_points.append(self.vertices[0])
            self.representative_points.append(self.vertices[0])
            return
        self.representative_points = []
        self.best_scattered_points = []
        furthest_distance = 0
        rep0 = None
        rep1 = None    
        # print('Vertices')   
        # print(self.vertices)
        # find the best scattered points
        for v in self.vertices:
            distance = euclidean_distance(v, self.centroid)            
            if distance > furthest_distance:
                furthest_distance = distance
                rep0 = v 
        if rep0 == None:
            rep0 = self.vertices[0]
        best_scat0 = Vertex(rep0.x, rep0.y)   
        self.best_scattered_points.append(best_scat0)
        next_furthest_distance = 0
        for v in self.vertices:
            distance = euclidean_distance(v, best_scat0)
            if distance > next_furthest_distance:
                next_furthest_distance = distance
                rep1 = v
        if rep1 == None:
            rep1 = self.vertices[1]
        best_scat1 = Vertex(rep1.x, rep1.y)
        self.best_scattered_points.append(best_scat1)  
        # with the best scattered points make the representative points     
        # the distance vectors are the distances of the best scattered points from the centroid
        rep0_distance_vector = distance_vector(self.centroid, best_scat0)
        rep1_distance_vector = distance_vector(self.centroid, best_scat1) 
        # reduce the vectors to be a fraction of what they were
        rep0_distance_vector = reduce_by_scalar_value(rep0_distance_vector)
        rep1_distance_vector = reduce_by_scalar_value(rep1_distance_vector)        
        rep_point0 = Vertex()
        rep_point1 = Vertex()        
        # use the new vectors to create the representative points
        rep_point0.x = best_scat0.x + rep0_distance_vector['x']
        rep_point0.y = best_scat0.y + rep0_distance_vector['y']
        rep_point1.x = best_scat1.x + rep1_distance_vector['x']
        rep_point1.y = best_scat1.y + rep1_distance_vector['y']        
        self.representative_points.append(rep_point0)
        self.representative_points.append(rep_point1)
        
class Entry:
    def __init__(self, key, val):
        self.key = key
        self.value = val

# i had to make my own map data structure 
# because the default map functionality of python wasn't what I was looking for
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

def distance_vector(v1, v2):
    vec2 = {'x': 0, 'y': 0}
    vec2['x'] = v1.x - v2.x
    vec2['y'] = v1.y - v2.y
    return vec2

def reduce_by_scalar_value(vec2):
    default_shrink_factor = 0.2
    vec2['x'] *= default_shrink_factor
    vec2['y'] *= default_shrink_factor
    return vec2    

# draws a graph with x and y values
# because this uses an actual dataset, data values in blue are male and red are female
# the x values is age and y values are income
def draw_cure_graph(clusters):
    x, y = get_raw_xy(clusters)

    fig, ax = plt.subplots()

    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    
    for cluster in clusters:
        _x = cluster.centroid.x
        _y = cluster.centroid.y
        circle = plt.Circle((_x, _y), cluster.radius, color='blue', fill=False)
        ax.add_patch(circle)

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

# returns 100 predefined values
def static_vertices():
    x = [71, 11, 52,  1, 92, 36, 98, 52, 14, 54, 83, 46, 11, 96, 76, 54, 24, 19, 31, 49, 43, 84, 80, 79,
    65, 92, 58, 74, 73, 56, 98, 98, 55, 37,  6, 94,  9, 31, 31, 59, 47, 88, 88, 51, 26, 44, 51, 41,
    94, 66,  2, 36, 49, 27, 31, 58, 78, 29, 12, 73,  4, 19, 36, 23, 32, 72, 83, 38, 97, 80, 74, 68,
    19, 96, 88, 38, 65, 42, 94, 85, 24,  7, 42, 47, 74, 48, 49,  1, 85,  5, 23, 71, 49, 41, 28, 77,
    67, 51, 69, 53]
    y = [34, 10, 39, 39, 39, 64, 27,  6, 10, 29, 37, 24, 33, 60, 31, 86, 89, 55, 10, 12,  8, 47, 45, 15,
    60,  1, 64, 37, 21, 65, 58, 18, 34, 80, 47, 87, 18, 32, 42, 56, 98, 65, 83, 54, 29, 35, 54, 93,
     6,  8, 39, 97, 45, 86, 94, 90, 28, 47, 43, 43, 58, 86, 46, 34,  3, 53, 20, 81, 80, 94, 57, 71,
    63, 83, 86, 50, 63,  9, 98, 18, 24, 53, 34, 54, 13, 65, 40, 53, 18,  1, 60, 32, 52, 62,  2,  3,
    33, 28, 53, 96]
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
    
def one_cluster_test_vertices(count):
    y_high = 75
    y_low = 25
    x_high = 75
    x_low = 25
    x = np.random.randint(low=x_low, high=x_high, size=count)
    y = np.random.randint(low=y_low, high=y_high, size=count)
    return x, y

# this is the main clustering algorithm
def cure_clustering(desired_cluster_count, x_data, y_data): 
    
    # x, y = random_vertices(num_vertices)    
    # clusters = []
    # for i in range(num_vertices):
    #     v = Vertex(x[i], y[i])
    #     cluster = Cluster([v])        
    #     cluster.calc_representative_points()
    #     clusters.append(cluster)
    clusters = []
    #print(data_frame)
    for i in range(len(x_data)):                
        v = Vertex(x_data[i], y_data[i])
        cluster = Cluster([v])
        cluster.calc_representative_points()        
        clusters.append(cluster)
    num_clusters = len(clusters)    

    # this is the main loop of the algorithm
    # it works very similarly to regular hierarchical clustering but instead of using 
    # the centroid of the neighboring cluster, it calculates a distance using its representative points
    # it puts that pair into the cluster map and proceeds as normally
    while num_clusters > desired_cluster_count:        
        c_map = ClusterMap()
        for i in range(len(clusters)):
            nearest_neighbor_distance = math.inf
            for j in range(len(clusters)):
                if i != j:  # so a cluster doesn't consider itself to be its nearest neighbor
                    for p in clusters[j].representative_points:  # loop over the representative points of the neighboring cluster                      
                        distance = euclidean_distance(clusters[i].centroid, p)
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
        for c in clusters:
            c.calc_representative_points()
    draw_cure_graph(clusters)
    
def main():
    desired_cluster_count = 20
    data_frame = pd.read_csv('./data/countries of the world.csv')    
    x, y = random_vertices(100)
    
    print(x)
    print(y)
    # cure_clustering(desired_cluster_count, x, y)

if __name__ == "__main__":
    main()