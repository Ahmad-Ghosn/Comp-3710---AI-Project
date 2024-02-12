# Patrick Carnevale, Alex Biru, Ahmad Ghosn, ALi Ghosn

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')

X = data.drop(['date', 'Appliances'], axis=1)   # X is a variable of the data dataframe without the date and appliances.
                                                #  The axis specifies which columns to drop
Y = data['Appliances']                          # The Y is a variable that containes the appliance column from the data dataframe


# X is the variable that will be used to make predictions
# Y is the variable that we want to predict
# train_test_split splits the datase tinto a training set and a testing set

# test_size specifies the portion of the dataset that we will use for testing, the rest is used for training
# eg. test_size = 0.1 (10% of the data is used for testing and 90% will be used for training)

# random_state is used to ensure that the split is reproducible

#X-train is training set, X-test is testing set
#Y-train is training set target, Y-test is testing set target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# Creates an instance of a multilayer perceptron neural network
# hidden_layer_sizes specifies the number of neurons in each hidden layer of the neural network
# hidden_layer_sizes(50, 50) means that there are 2 hidden layers with 50 neurons each

# activation specifies the activation function used in the hidden layers.
# activation = 'relu' means that the Rectified Linear Unit activation fuction is used

# solver specifies the optimization algorithm used to update the weights of the neural network
# max_iter is the max number of iterations that the solver should run.
# random_state is used to ensure that the results of the neural network are reproducable
clf = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
