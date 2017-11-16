import math
import random
import numpy as np
import scipy
import lvq_exceptions as exc
import knn
from scipy.spatial.distance import minkowski
from knn import  get_nearest_neighbour

class Lvq1:
    """
    Class for creating, training and testing a lvq neural network.

    Attributes:
        neurons_per_class: list with the number of neurons (or codebook vectors)
            the neural network model will use. The total number of neurons must be
            a multiple of the classes used in the classification problem.
        neuron_count: total number of neurons used in the network.
        class_percentages: list of typical class percentages based on the neurons per 
            class given as input. In a binary classification problem with a balanced
            dataset the percentages would be [0.5, 0.5]
        class_labels: lists with labels used for each class. For example, [0, 1]
            means that the first class is represented by the label 0 while the second
            class is represented by 1.
        learning_rate: float variable that determines how fast the neurons are
            learning (i.e how large the corrections will be on the neuron weights)
        epochs: integer that determines how many times the training examples will be
            fed to the network for training.
        class_count: number of classes that will be used for classification
        neuron_weights: a matrix with the weights of each neuron where each row
            represents a neuron
        neuron_labels: a 1xn vector where n is the number of neurons and the value
            of each element represents the class label of the neuron.
        early_stopping: a boolean variable which is True when early stopping is being used
            and false when it is not. Early stopping is being implemented with a validation
            set which is part of the provided training data.
        validation_set_perc: percentage of training data set that will be used for validation
            if early_stopping is True and no validation set is given.
        validation_set: a set used for early stopping when early_stopping is True.
        validation_labels: labels used for validation set
        validation_set_rdy: boolean variable that is True when there's a validation set available
            for use by the model (i.e when validation_set variable is set).
        patience: determines for how many epochs the accuracy must fall in order to make 
            an early stop.
    """

    def __init__(self, neurons_per_class, class_labels, epochs = 30, learning_rate =  0.01, patience = 5, validation_set_perc = 0.1):
        """
        Initializes neural network model with given parameters. Raises an InvalidParameterException
            if a parameter value is not valid.
        """
        
        #check if at least 2 classes will be used for the classification problem
        if(len(neurons_per_class) <= 1):
            raise exc.InvalidParameterException("\n[Exception]: You must use at least 2 classes to train the lvq network." \
                " Provide a list with the neuron counts for each class as the first parameter.\n")
        
        self.class_count = len(neurons_per_class)
        
        #check if the number of class labels is the same as the number of classes
        if(self.class_count != len(class_labels)):
            raise exc.InvalidParameterException("\n[Exception]: Invalid parameter provided. The number of class labels is " \
                "not the same as the number of classes.\n")
        
        invalid_neuron_counts = sum(1 for neurons in neurons_per_class if neurons < 0 or type(neurons) is not int)

        if(invalid_neuron_counts > 0):
            raise exc.InvalidParameterException("\n[Exception]: Neuron counts cannot take negative values or be any type other than integers.\n")
        elif(epochs < 0):
            raise exc.InvalidParameterException("\n[Exception]: Number of training epochs cannot take negative values.\n")
        elif(learning_rate <= 0 or learning_rate >= 1):
            raise exc.InvalidParameterException("\n[Exception]: Learning rate takes values only between 0 and 1.\n")
        elif(validation_set_perc <= 0 or validation_set_perc >=1):
            raise exc.InvalidParameterException("\n[Exception]: Invalid percentage of validation set.")

        self.neurons_per_class = neurons_per_class
        self.neuron_count = sum(neuron_count for neuron_count in neurons_per_class)
        self.class_labels = class_labels
        self.class_percentages = [neurons_in_class / float(self.neuron_count) for neurons_in_class in self.neurons_per_class]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = False
        self.validation_set_perc = validation_set_perc
        self.validation_set_rdy = False
        self.patience = patience
        
        
    def train(self, training_data, target_labels, weight_init_method = "real-samples", k = 3, p = 2):
        """
        Initializes the weights for the neural network and trains it. 

        Args:
            training_data: qxn matrix that represents 'q' n-dimensional input vectors.
            target_labels: qx1 vector that represent the target class label (correct output)
                for the q n-dimensional vectors in <training_data>.
            weight_init_method: determines the weight-initialization method to be used. The default
                method for weight initialization uses real samples as initial weights which are
                first checked against their k-nearest neighbours to see if they are misclassified.
                If most samples are found to be misclassified by knn then random weights are used.
            k: determines the k in knn-classification that will be used to determine the initial
                weights of the neurons.
            p: determines the p in minkowski distance which is used as a metric in the neural
                network model. The default value of p is 2 for which the Minkowski distance is
                equal to the Euclidean distance.
        """
        np.random.seed(1)
        
        print("\n    Initializing the weights of the lvq network...\n")
        if(weight_init_method == "real-samples"):
            self.__real_sample_initialization(training_data, target_labels, k, p)
        elif(weight_init_method == "random"):
            self.__random_weight_initialization(training_data.shape[1])
        
        print("    Initialization finished. Starting the neural network training...\n")
        
        print("    Shuffling the training dataset...\n")
        data = self.__shuffle_data(training_data, target_labels)
        training_data = data[0]
        target_labels = data[1]

        #if early stopping is activated and a validation set is not provided
        if(self.early_stopping == True and (not self.validation_set_rdy)):
                print("    Seperating training data into training and validation data...\n")
                samples_per_class = self.__get_samples_per_class(training_data)
                seperated_dataset_tuple = self.__seperate_dataset(training_data, target_labels, samples_per_class, self.validation_set_perc)
                training_set = seperated_dataset_tuple[0][0]
                training_labels = seperated_dataset_tuple[0][1]
                self.validation_set = seperated_dataset_tuple[1][0]
                self.validation_labels = seperated_dataset_tuple[1][1]
        else:
            training_set = training_data
            training_labels = target_labels
        

        print("    Starting training...\n")
        learning_rate = self.learning_rate
        sample_count = training_set.shape[0]
        lr_max_iterations = float(sample_count * self.epochs)
        max_accuracy_params =  (-1 ,np.copy(self.neuron_weights)) #initialize for -1 accuracy and starting weights
        max_accuracy_epoch = -1
        lower_accuracy_steps = 0

        for i in range(self.epochs):
            print("    Epoch " + str(i))

            for index,example in enumerate(training_set):
                nearest_neighbour_index = get_nearest_neighbour(example, self.neuron_weights)[0] 
                nearest_neighbour_weights = self.neuron_weights[nearest_neighbour_index]
                nearest_neighbour_label = self.neuron_labels[nearest_neighbour_index]
                
                #if label is correct
                if(nearest_neighbour_label == training_labels[index]):
                    nearest_neighbour_weights += learning_rate * (example - nearest_neighbour_weights)
                else:
                    nearest_neighbour_weights -= learning_rate * (example - nearest_neighbour_weights)
                
                self.neuron_weights[nearest_neighbour_index] = nearest_neighbour_weights

                learning_rate = self.learning_rate - self.learning_rate * ((i * sample_count + index) / lr_max_iterations)

            if(self.early_stopping == True):
                accuracy = self.test(self.validation_set, self.validation_labels)
                if(accuracy > max_accuracy_params[0]):
                    lower_accuracy_steps = 0
                    max_accuracy_epoch = i + 1
                    max_accuracy_params = (accuracy, np.copy(self.neuron_weights))
                else:
                    lower_accuracy_steps += 1

                if(lower_accuracy_steps >= self.patience or i == (self.epochs - 1)):
                    self.neuron_weights = max_accuracy_params[1]
                    return (max_accuracy_params[0], max_accuracy_epoch) #return max validation accuracy achieved in training and epoch when the training stopped
     


    def test(self, test_data, test_labels):
        """
        Runs the neural network with the test_data given the test_labels
            and calculates the accuracy of the model.
        """
        
        correct_labels = 0
        for index, sample in enumerate(test_data):
            predicted_label = self.predict_label(sample)
            if(test_labels[index] == predicted_label):
                correct_labels += 1           

        return correct_labels / float(test_data.shape[0])


    def predict_label(self, sample):
        """
        Predicts label for a given sample.

        Args:
            sample: vector whose label will be predicted.
    
        Returns:
            Predicted label.
        """
        nearest_neighbour_index = get_nearest_neighbour(sample, self.neuron_weights)[0]
        return self.neuron_labels[nearest_neighbour_index]


    def set_early_stopping(self, bool_var):
        """
        Setter for the early_stopping class attribute.

        Raises:
            InvalidParameterException: raises this exception if the bool_var
                parameter is not a boolean variable.
        """
        #check if the parameter is not a boolean
        if(not isinstance(bool_var, bool)):
            raise exc.InvalidParameterException("\n[Exception]: The bool_var parameter of set_early_stopping" \
                " must be a boolean.\n")
        self.early_stopping = bool_var


    def set_validation_set(self, validation_set, validation_labels):
        self.validation_set = validation_set
        self.validation_labels = validation_labels
        self.validation_set_rdy = True

    def __real_sample_initialization(self, training_data, target_labels, k, p):
        """
        Initializes the weights for the neural network by using real data samples
            as initial weights. The samples of which the values are going to be used as
            a neuron's initial weights are checked to make sure they are not misclassified
            against their k-nearest neighbours. If knn classification is not viable in the
            dataset then the weights are initialized randomly.
        """
        
        neuron_counts =  [neuron for neuron in self.neurons_per_class]
        weights_uninitialized = sum([neuron for neuron in neuron_counts])
        class_count = self.class_count
        neuron_weights = []
        neuron_labels = []
        rows = training_data.shape[0]
        
        #for each vector
        for i in range(rows):
            #for each label
            for j in range(class_count):
                #find the index of the label value in the label list
                if(target_labels[i] == self.class_labels[j]):
                    #if more neurons are needed for this class
                    if(neuron_counts[j] > 0):
                        #get label from the majority of knn of training_data[i]
                        knn_label = knn.knn_classify(training_data[i], training_data, target_labels, self.class_labels, k)   
                        #if the majority of the k-nearest-neighbours have the same class as the true class of the training data sample
                        if(knn_label == target_labels[i]):
                            #Initialize neuron weights based on the training data sample
                            neuron_weights.append(training_data[i])
                            neuron_labels.append(target_labels[i])
                            weights_uninitialized -= 1
                            neuron_counts[j] -= 1
                    break

        
        #check if all weights were initialized
        if(weights_uninitialized != 0):
            print("Using real samples as initial weights was not possible. Random initial weights will be used...\n")
            self.__random_weight_initialization(training_data.shape[1])
        else:
            self.neuron_weights = np.array(neuron_weights)
            self.neuron_labels = np.array(neuron_labels)

            
    def __random_weight_initialization(self, dimensions):
        """
        Initializes neuron weights with random values from 0 to 1.

        Args:
            dimensions: determines the dimensions of the neuron weight vectors
        """
  
        neuron_counts = [neuron for neuron in self.neurons_per_class]
        total_neurons = sum([neuron for neuron in neuron_counts])
        neuron_weights = []
        neuron_labels = []
        
        #for each neuron count of each class
        for index, neuron_count in enumerate(neuron_counts):
            #add <neuron_count> neurons of this class to the lvq network
            for i in range(neuron_count):
                neuron_weights.append(np.random.rand(1, dimensions)[0])
                neuron_labels.append(self.class_labels[index])
        
        self.neuron_weights = np.array(neuron_weights)
        self.neuron_labels = np.array(neuron_labels)
        
    def __get_samples_per_class(self, training_data):
        """
        Gets the number of training samples used for each class.

        Args:
            training_data: qxn matrix with the training data.

        Returns:
            A list with the counts of training samples used for each class.
        """
        sample_count = training_data.shape[0]

        samples_per_class = []
        s = sample_count
        for i in range(self.class_count):
            if(i != (self.class_count - 1)):
                samples = math.ceil(self.class_percentages[i] * sample_count)
                samples_per_class.append(samples)
                s -= samples
            else:
                samples_per_class.append(s)

        return samples_per_class



        
    def __seperate_dataset(self, training_data, target_labels, samples_per_class, validation_set_perc):
        """
        Seperates the training_data variable into data for training and validation.
        
        Args:
            training_data: qxn matrix with the training data samples
            target_labels: qx1 matrix with the target labels for each training data
                sample
            samples_per_class: number of samples used for each class in training data
            validation_set_perc: determines the percentage of training_data
            that will be used for validation.

        Returns:
            A list of 2 tuples. The first tuple has the matrix and the target labels
                for the training data while the second tuple has the matrix and the target 
                labels for the validation data.
        """
        total_sample_count = training_data.shape[0]
    
        #calculate the number of samples that will be used for each class in the validation step
        val_samples_per_class = [math.ceil(validation_set_perc * sample_count) for sample_count in samples_per_class]
 
        #turn training data and its labels to lists
        tr_data_list = training_data.tolist()
        label_list = target_labels.tolist()
       
        training = []
        validation = []
        training_labels = []
        validation_labels = []

        for i in range(total_sample_count):
            for j in range(self.class_count):
                if(target_labels[i] == self.class_labels[j]):
                    if(val_samples_per_class[j] > 0):
                        val_samples_per_class[j] -= 1
                        validation.append(training_data[i])
                        validation_labels.append(target_labels[i])
                    else:
                        training.append(training_data[i])
                        training_labels.append(target_labels[i])

        #convert lists to matrices
        training = np.array(training)
        validation = np.array(validation)
        training_labels = np.array(training_labels)
        validation_labels = np.array(validation_labels)
        return [(training, training_labels), (validation, validation_labels)]
        
    def __shuffle_data(self, samples, targets):
        """
        Shuffles the samples and their target outputs and returns a tuple with the
            shuffled data.

        Args:
            samples: qxn numpy array
            targets: qx1 numpy array

        Retuns:
            A tuple with the samples and the targets shuffled. The tuple has the
                form (samples, targets)
        """
        data = zip(samples.tolist(), targets.tolist())
        np.random.shuffle(data)
        data = zip(*data)

        return (np.array(data[0]), np.array(data[1]))
