import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

#TODO
# plot silhouette analysis


def eucl_sq_dist(point1, point2):
    """
    Computes the squared Euclidean distance
    between two points
    
    Arguments
        point1, point2 : (list)
        
    Returns
        (float) squared Euclidean distance
    """
    
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    return np.sum(np.square(point1 - point2))


class KMeans:
    """
    This class implements the KMeans algorithm
    with random initialization of the centroids
    """
    
    def __init__(self, K):
        self.K = K
        

    def rand_centroids(self, data):
        """
        Picks K random points out of the
        pd dataframe 'data'
        """

        return data.sample(self.K).values

    
    def dist_to_centroids(self, point):
        """
        Returns an array the squared Euclidean 
        distance between 'point' and each centroid
        """
        
        return [eucl_sq_dist(point, c) 
                for c in self.centroids]
        
        
    def assign_labels(self, data):
        """
        Returns a list with the indexes of 
        the closest centroid to each point 
        """

        return np.array([np.argmin(self.dist_to_centroids(point)) 
                         for point in data.values])
    
    
    def compute_centroids(self, data):
        """
        Given a certain set of clusters,
        this function computes the true centroid
        for each cluster
        """
        
        true_centroids = []
        
        for c in range(len(self.centroids)):
            points = data.iloc[self.labels == c]
            true_centroids.append(np.sum(points) / len(points))
            
        return true_centroids
      

    def has_converged(self, new_centroids):
        """
        Returns True if the algorithm has converged,
        i.e. if in the next iteration of the algorithm
        the centroids haven't changed
        """
        
        return np.array_equiv(self.centroids, new_centroids)
    
    
    def compute_inertia(self, data):
        """
        Computes the Sum of Squared Errors (SSE),
        i.e. the sum of the distances of each point
        to its nearest centroid
        """
        
        inertia = 0
        
        for idx, c in enumerate(self.centroids):
            points = data.iloc[self.labels == idx]
            inertia += np.sum([eucl_sq_dist(point, c) 
                               for point in points.values])
                               
        return inertia
    
    def compute_silhouette(self, data):
        """
        Computes the silhouette score for each point,
        defined as
        
        s = (b - a) / min(a, b)
        
        where a is the mean distance between a point
        and the others in the same cluster
        and b is the minimum distance among 
        the mean distances between a point 
        and all points in another cluster
        
        Finally, it returns the average score
        over all points
        """
        
        if self.K == 1:
            return 1
        
        s = np.empty(len(data))
        
        for idx, point in enumerate(data.values):
            
            # all points in the same cluster of 'point'
            cluster_points = data.iloc[self.labels == self.labels[idx]]
            
            # if 'point' is the only point in its cluster
            # it has a score of 0
            if len(cluster_points) == 1:
                s[idx] = 0
                continue
            
            # mean distance between 'point' and all the others
            # in the same cluster
            a = np.sum([np.sqrt(eucl_sq_dist(point, cp)) 
                       for cp in cluster_points.drop(idx).values])
            
            a = a / (len(cluster_points) - 1)
            
            # mean distances of 'point' to the points
            # of each other cluster
            b = []            
            for label in filter(lambda k: k != self.labels[idx], range(self.K)):
            
                other_points = data.iloc[self.labels == label]
                
                b_k = np.sum([np.sqrt(eucl_sq_dist(point, op))
                                 for op in other_points.values])
                
                b.append(b_k / len(other_points))
                
            b_min = np.min(b)
            
            # silhouette score for 'point' 
            s[idx] = (b_min - a) / (np.max([a, b_min]))
            

        return np.mean(s)
    

    def fit(self, data):
        """
        When called this method starts the KMeans algorithm,
        which will stop when convergence has been reached
        """

        # the initial centroids are chosen randomly
        self.centroids = self.rand_centroids(data)
        self.labels = self.assign_labels(data)
        
        while True:

            new_centroids = self.compute_centroids(data)

            if not self.has_converged(new_centroids):
                self.centroids = new_centroids  
                self.labels = self.assign_labels(data)
            else:
                self.inertia = self.compute_inertia(data)     
                break
                

    def display(self, x, y, show_centroids = False):
        """
        Shows a scatter plot with the clusters
        obtained
        """
        
        plt.scatter(x, y, c = self.labels, edgecolors = "b")
        
        if show_centroids:
            
            x_c = np.transpose(self.centroids)[0]
            y_c = np.transpose(self.centroids)[1]
            
            plt.scatter(x_c, y_c, 
                        color = "green", 
                        marker = "X", s = 80,
                        edgecolors = "w")
        
        plt.show()


def elbow(data, K_list, plot = False):
    """
    Runs KMeans for different numbers of clusters
    and computes the inertia (SSE) for each
    """
    
    inertia = []
    
    for k in K_list:
        kmeans = KMeans(k)
        kmeans.fit(data)
        inertia.append(kmeans.inertia)
        
    if plot:
        plot_elbow(K_list, inertia)
        
    return inertia


def plot_elbow(K_list, inertia):
    """
    Creates a plot of the inertias obtained
    for different number of clusters
    """
    
    plt.title("KMeans SSE (Inertia)")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    
    plt.plot(K_list, inertia, marker = "o")
    
    plt.xticks(ticks = K_list, labels = K_list)
    
    plt.vlines(x = K_list, 
               ymin = [0]*len(K_list), ymax = inertia,
               linestyles = "dotted")


def silhouette(data, K_list, plot = False):
    """
    Runs KMeans for different numbers of clusters
    and computes the inertia (SSE) for each
    """
    
    silh = []
    
    for k in K_list:
        kmeans = KMeans(k)
        kmeans.fit(data)
        silh.append(kmeans.compute_silhouette(data))
        
    if plot:
        plot_silhouette(K_list, silh)
        
    return silh


def plot_silhouette(K_list, silh):
    """
    Creates a plot of the inertias obtained
    for different number of clusters
    """
    
    plt.title("Average Silhouette")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette")
    
    plt.plot(K_list, silh, marker = "o")
    
    plt.xticks(ticks = K_list, labels = K_list)
    
    # plt.vlines(x = K_list, 
    #            ymin = [0]*len(K_list), ymax = silh,
    #            linestyles = "dotted")