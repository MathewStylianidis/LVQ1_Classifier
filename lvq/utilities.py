import lvq_exceptions
from lvq_exceptions import EmptyListException

def get_max_index(x):
    """
	Args:
	    x: a list of elements

    Returns:
        Returns the index of the element with the maximum value in x.
        
	Raises:
	    EmptyListException: This exception is raised if the list x is empty.
    """
	#if list is empty
    if(not x):
        raise EmptyListException("[Exception]: Cannot find the maximum value of an empty list.")

    max_value = x[0]
    max_index = 0
    for ind, value in enumerate(x):
        if(value > max_value):
            max_value = value
            max_index = ind
    
    return max_index
	    



