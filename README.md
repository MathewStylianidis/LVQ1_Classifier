# LVQ1 Classifier
Implementation of a learning vector quantization classifier in python 2.7. 

The LVQ classifier can be used by importing the class Lvq1 from the lvq1.py file.


## Required libraries
* SciPy
* NumPy
* nltk (English stopwords, Word tokenizer)

## How to use 
This is an example of how to use the LVQ model in code. 
```python

from lvq.lvq1 import Lvq1
from lvq.lvq_exceptions import InvalidParameterException

training_data , target_labels = getTrainingData() # Get list of training samples and label list
validation_data, validation_labels = getValidationData() # Get list of validation samples and label list


neuron_counts = [5, 5] # Number of neurons per class
labels = [0, 1] # Label for each class
max_epochs = 100
patience = 5 # Number of epochs to wait before stopping training
init_lr = 0.1 # Initial learning rate

try:
  # Create model object
  lvq_network = Lvq1([5,5], [0, 1], max_epochs, lr_hyperparam, patience)
  lvq_network.set_early_stopping(True)
  
  # Manually set validation instead of randomly splitting the training set
  lvq_network.set_validation_set(validation_data, validation_labels)
  
  # Train by initializing neuron weights with values of samples in the training data
  lvq_network.train(training_data, target_labels, "real-samples")  
  
except InvalidParameterException as exception:
  print(exception)
```


