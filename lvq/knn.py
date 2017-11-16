import numpy as np
import scipy
import operator
import utilities as util
import lvq_exceptions as exc
from scipy.spatial.distance import minkowski
from operator import itemgetter

def get_knn(vector, vectors, k, p = 2):
    """
    Gets a list of tuples of the k-nearest-neighbours indices in <vectors> parameter.

    Args:
        vector: vector: vector to be classified according to its knn in <vectors>
        vectors: qxn matrix that represents 'q' n-dimensional vectors that might
            be included in the list of k-nearest-neighbours of vectors[i].
        k: determines the k in knn (i.e the number of neighbours to be found)
        p: metric being used to calculate distance

    Returns:
        List of tuples with the neighbour indices along with their distances

    Raises:
        InvalidParameterException: Raised when a parameter has an invalid value
            that prohibits the correct function execution.

    """
    #if there are less than k vectors available raise an InvalidDimensionsException
    if(vectors.shape[0] < k):
        raise exc.InvalidParameterException("[Exception]: there are less than k vectors" \
        	"(rows) in the matrix parameter which was passed into get_knn_indices().")
    
    if(vector.shape[0] != vectors.shape[1]):
        raise exc.InvalidParameterException("[Exception]: vector parameter must have the same 'n'"\
            "dimensions as in the qxn <vectors> matrix.")

    #check if k is not positive
    if(not(k > 0)):
    	raise exc.InvalidParameterException("\n[Exception]: Invalid parameter provided. The parameter k must " \
        		"be positive.\n")

    #create a list containing the distance between vectors[i] and each other vector in <vectors> except for itself
    neighbours = []
    for index, v in enumerate(vectors):
        distance = minkowski(vector, v, p)
        neighbours.append((index, distance))
    
    neighbours.sort(key=itemgetter(1))

    return [neighbour for (index,neighbour) in enumerate(neighbours) if index < k]

def get_nearest_neighbour(x, vectors, p = 2):
    """
        Finds the vector in <vectors> with the minimum distance from the vector parameter x.

        Args:
            x: 1xn vector
            vectors: qxn matrix with vectors that are nearest neighbour candidates of vector x.
            p: distance metric used. Default is 2 (Euclidean distance).

        Returns:
            Tuple with the index of the nearest neighbour of x in vectors and the distance between
            the nearest neighbour and x.

        Raises:
            InvalidParameterException: Raised if x and each vector in vectors have different dimensions.
    """
    
    if(x.shape[0] != vectors.shape[1]):
        raise exc.InvalidParameterException("[Exception]: vector parameter x must have the same 'n'"\
            "dimensions as in the qxn <vectors> matrix.")
    
    minimum_distance = minkowski(x, vectors[0], p)
    minimum_index = 0
    for index, v in enumerate(vectors):
        distance = minkowski(x, v, p)
        if(minimum_distance > distance):
            minimum_distance = distance
            minimum_index = index
    #print(minimum_index)
    return (minimum_index, minimum_distance)


def knn_classify(vector, vectors, labels, class_values, k, p = 2):
    """
    Classifies vector according to the labels of its k-nearest-neighbours
        in <vectors>. The classification will be based on the label of the majority
        of the neighbours. If there is a tie then classification for k = 1 will take
        place.

    Args: 
        vector: vector to be classified according to its knn in <vectors>
        vectors: qxn matrix that represents 'q' n-dimensional vectors based on
            which the knn classification will take place.
        labels: qx1 array with the label of each of the q vectors in <vectors>.
        class_values: list of values that are used for each class (i.e [0 1] if 0
            is used for the first class and 1 for the second one).
        k: determines the k in knn classification.
        p: metric be used to calculate the Minkowski distance. When p = 2 the Minkowski
            distance is equal to the Euclidean distance and when p = 1 it is equal to
            the Manhattan distance.

    Returns:
        The class label of <vector> based on knn-classification
    """

    #get a list of the indices of the top k-nearest-neighbours along with the distances
    k_nearest_neighbours = get_knn(vector, vectors, k)
    
    #get the class of the majority of the k-nearest-neighbours
    class_counts = [0 for x in class_values]
    for neighbour in k_nearest_neighbours:
        neighbour_index = neighbour[0]
        #find the index of the neighbour class
        class_index = -1
        for ind, value in enumerate(class_values):
            if(value == labels[neighbour_index]):
                class_index = ind
        if(class_index != -1):
            class_counts[class_index] += 1

    majority_index = util.get_max_index(class_counts)
    return class_values[majority_index]
        
