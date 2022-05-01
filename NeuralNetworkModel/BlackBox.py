"""
    CS 410
    Author: Francisco Benavides
"""

import numpy as np
import tensorflow as tf
import seaborn
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

"""
 Id recommend moving this file into the AIProteins folder
"""
#from sqlite3 import adapt
#import Disulfide_Parser


"""
    This is a standard Dense Layer classification model.
"""



def load_data():

    # loading file and dataset
    """
    Call PDB parser here
    """

    data = Disulfide_Parser.adapter()
    
    features = np.copy(data[:, 0:4])
    labels = np.copy(data[:, 4])
    
    return [features, labels]

# normalize the dataset 
# X = (x[i] - mean) / standard_diviation
def feature_scaling(dataset):
    data = dataset[0]
    print(len(data))
    print(data.shape)
    for i in range(len(data)):
        mean = np.mean(data[i, :])
        sd = np.std(data[i, :])
        data[i,:] = (data[i,:] - mean) / sd

        #mean = np.mean(data[:, i])
        #sd = np.std(data[:, i])
        #data[:,i] = (data[:,i] - mean) / sd
    
    dataset = [data, dataset[1]]
    return dataset


# split dataset
# 10% testing, 18% validation, and 72% training
def dataset_split(data, labels):
    # split the data into train, validation, and test
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.9, test_size=0.1, shuffle=True)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, shuffle=True)

    return [x_train, x_validate, x_test, y_train, y_validate, y_test]




"""

    Utility Class and Functions

"""

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def __init__(self):
       self.training_batch_log = [[],[]]
       self.testing_batch_log = [[],[]]

    def returnTraining(self):
        return self.training_batch_log
    
    def returnTesting(self):
        return self.testing_batch_log

    def on_train_batch_end(self, batch, logs=None):
        #print(
        #    "Up to batch {}, the average accuracy is {:7.2f}, the average loss is {:7.2f}.".format(batch, logs["accuracy"], logs["loss"])
        #)
        self.training_batch_log[0].append(logs["accuracy"])
        self.training_batch_log[1].append(logs["loss"])

    def on_test_batch_end(self, batch, logs=None):
        #print(
        #    "Up to batch {}, the average accuracy is {:7.2f}, the average loss is {:7.2f}.".format(batch, logs["accuracy"], logs["loss"])
        #)
        self.testing_batch_log[0].append(logs["accuracy"])
        self.testing_batch_log[1].append(logs["loss"])


# helper function used to generate ROC curve and check accuracy.
def compare_results(raw_predictions, y_test):
    y_predicted = []

    correct = 0
    total = len(raw_predictions)

    for i in range(len(raw_predictions)):
        j = np.where(raw_predictions[i,:] == max(raw_predictions[i,:]))
        #x = [0., 0., 0.] #np.zeros(3)
        x = [0., 0.]
        x[j[0][0]] = 1.
        if np.array_equal(y_test[i,:], x):
            correct += 1
        y_predicted.append(x)
    y_predicted = np.array(y_predicted)
    
    wrong = total - correct

    return [y_predicted, y_test, correct, wrong, raw_predictions]

def util_helper(y_pred, y_test):
    new_y_test = []

    for i in range(len(y_pred)):
        #if np.array_equal(y_test[i,:], np.array([1, 0, 0])):
        if np.array_equal(y_test[i,:], np.array([1, 0])):
            new_y_test.append(0)
            
        #if np.array_equal(y_test[i,:], np.array([0, 1, 0])):
        if np.array_equal(y_test[i,:], np.array([0, 1])):
            new_y_test.append(1)
            
        """
        if np.array_equal(y_test[i,:], np.array([0, 0, 1])):
            new_y_test.append(2)
        """

    return new_y_test




"""

    Neural Network Model

"""


def neural_network(data, batchNormalize=True, learning_rate=0.000001, batch_training=False, activation_function='ReLU'):

    # load the data in.
    x_train = data[0]
    x_validate = data[1]
    x_test = data[2]

    y_train = data[3]
    y_validate = data[4]
    y_test = data[5]



    # Hyper parameters:

    if batch_training:
        num_epochs = 1
        batch_size = 1
    else:
        num_epochs = 25
        batch_size = 100

    eta = learning_rate
    
    decay_factor = 0.95

    size_hidden = 500 # nodes per layer

    # static parameters
    size_input = 4 # number of features
    size_output =  2 # number of labels
    Input_shape = (size_input,)

    # model:

    if batchNormalize:

        # model with batch normalization
        model = Sequential([
            keras.Input(shape=Input_shape, name='input_layer'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer01'),
            BatchNormalization(),
            Dense(size_hidden, activation=activation_function, name='hidden_layer02'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer03'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer04'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer05'),
            
            # Dense(size_hidden, activation=activation_function, name='hidden_layerXX'),
            
            Dense(size_output, activation='softmax', name='output_layer')])
    else:
        # model without batch normalization
        model = Sequential([
            keras.Input(shape=Input_shape, name='input_layer'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer01'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer02'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer03'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer04'),
            Dense(size_hidden, activation=activation_function, name='hidden_layer05'),
            
            # Dense(size_hidden, activation=activation_function, name='hidden_layerXX'),
            
            Dense(size_output, activation='softmax', name='output_layer')])


    model.summary()


    # initializing label in vector form [1, 0, 0, ...], [0, 1, 0, ...], [0, 0, 1, ...]
    y_train_vectors = utils.to_categorical(y_train)
    y_test_vectors = utils.to_categorical(y_test)
    y_validate_vectors = utils.to_categorical(y_validate)

    #setting up optimizers
    # using Sarcastic Gradient Decent
    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=eta, decay_steps=x_train.shape[0], decay_rate=decay_factor)
    SGD_optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule) #keras.optimizers.SGD(learning_rate=learning_rate_schedule)
    
    # setting up model.
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=SGD_optimizer, metrics='accuracy')

    if batch_training: # we are training with a epoch of 1 and batch_size of 1 to obtain the learning curve.
        batch_learning_curve_info = LossAndErrorPrintingCallback()
        
        # history.history['accuracy']['val_accuracy']['loss']['val_loss']
        history = model.fit(x_train, y_train_vectors, batch_size=batch_size, epochs=num_epochs, validation_data=(x_validate, y_validate_vectors), verbose=2, callbacks=[batch_learning_curve_info])
    else:
        history = model.fit(x_train, y_train_vectors, batch_size=batch_size, epochs=num_epochs, validation_data=(x_validate, y_validate_vectors), verbose=2)

    # [loss, accuracy]
    results = model.evaluate(x_test, y_test_vectors, batch_size)

    # prediction
    predictions = model.predict(x_test)


    # Return this
    prediction_info = compare_results(predictions, y_test_vectors)
    
    correct = prediction_info[2]
    wrong = prediction_info[3]
    total = correct + wrong
    print(correct, wrong, total)

    # fit, evaluation, prediction
    if batch_training:
        return [history, results, prediction_info, batch_learning_curve_info.returnTraining(), batch_learning_curve_info.returnTesting()]
    
    return [history, results, prediction_info]




"""

    Graphing Functions

"""

def parameter_tuning(v_loss, t_loss, title):

    plt.plot(v_loss, "o", 1, color="red", label="Validation loss")
    plt.plot(t_loss, "o", color="blue", label="Training loss")

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()



def learning_curve(learn_info): # for batch learning
    plt.plot(learn_info[0])
    plt.plot(learn_info[1])
    plt.title('accuracy over samples')
    plt.xlabel("Samples")
    plt.ylabel("Cost")
    plt.legend(['train', 'loss'], loc='upper left')
    plt.show()
    
def roc_graph(prediction_info, extra=""):
    y_pred = prediction_info[4]
    y_test = prediction_info[1]
    results = util_helper(y_pred, y_test)

    fpr0, tpr0, thresholds0 = metrics.roc_curve(y_test[:, 0], y_pred[:, 0])
    auc0 = metrics.auc(fpr0, tpr0)
    plt.plot(fpr0, tpr0, color="red", lw=4, label="ROC curve of class {0} (area = {1:0.2f})".format(0, auc0))    

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test[:, 1], y_pred[:, 1])
    auc1 = metrics.auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, color="gold", lw=3, label="ROC curve of class {0} (area = {1:0.2f})".format(1, auc1))

    plt.title("ROC Graph " + extra)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

def multi_roc_graph(prediction_info1, prediction_info2, extra=""):
    y_pred1 = prediction_info1[4]
    y_test1 = prediction_info1[1]

    y_pred2 = prediction_info2[4]
    y_test2 = prediction_info2[1]

    # results = util_helper(y_pred1, y_test1)

    fpr0, tpr0, thresholds0 = metrics.roc_curve(y_test1[:, 0], y_pred1[:, 0])
    auc0 = metrics.auc(fpr0, tpr0)
    plt.plot(fpr0, tpr0, color="maroon", lw=4, label="Model 1 ROC curve of class {0} (area = {1:0.2f})".format(0, auc0))    

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test1[:, 1], y_pred1[:, 1])
    auc1 = metrics.auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, color="red", lw=3, label="Model 1 ROC curve of class {0} (area = {1:0.2f})".format(1, auc1))
    
    fpr0, tpr0, thresholds0 = metrics.roc_curve(y_test2[:, 0], y_pred2[:, 0])
    auc0 = metrics.auc(fpr0, tpr0)
    plt.plot(fpr0, tpr0, color="navy", lw=2, label="Model 2 ROC curve of class {0} (area = {1:0.2f})".format(0, auc0))    

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test2[:, 1], y_pred2[:, 1])
    auc1 = metrics.auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, color="blue", lw=2, label="Model 2 ROC curve of class {0} (area = {1:0.2f})".format(1, auc1))
    
    plt.title("ROC Comparison Graph " + extra)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

def confusion_matrix(prediction_info):
    y_pred = prediction_info[0]
    y_test = prediction_info[1]

    y_test_1d = util_helper(y_pred, y_test)
    y_pred_1d = util_helper(y_test, y_pred)

    cf_matrix = metrics.confusion_matrix(y_test_1d, y_pred_1d)

    ax = seaborn.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Reds')

    ax.set_title('Confusion Matrix\n')
    #ax.set_xlabel('\nPredicted Location')
    #ax.set_ylabel('Actual Location')

    # Ticket labels - List must be in alphabetical order
    #ax.xaxis.set_ticklabels(["""FILL IN"""])
    #ax.yaxis.set_ticklabels(["""FILL IN"""])

    # Display the visualization of the Confusion Matrix.
    plt.show()



def main():
    dataset = load_data()

    split_data = dataset_split(dataset[0], dataset[1])

    """
    NB = No Batch normalization
    YB = Yes Batch normalization

    NF = No Feature scaling
    YF = Yes Feature scaling

    01 = learning curve of .01
    25 = learning curve of .25
    """

    v_loss = []
    t_loss = []

    NBNF = neural_network(split_data, False) # No batch normalization and No feature scaling

    v_loss.append(NBNF[0].history['val_loss'][0])
    t_loss.append(1)
    t_loss.append(NBNF[0].history['loss'][0])


    
    YBNF = neural_network(split_data, True) # With batch normalization and No feature scaling
    v_loss.append(YBNF[0].history['val_loss'][0])
    t_loss.append(YBNF[0].history['loss'][0])



    """
    
    Models beyond this point have feature scaling enabled

    """
    feature_scaled_dataset = feature_scaling(dataset) # normalize the data with feature scaling formula                                                 # FIX
    split_data = dataset_split(feature_scaled_dataset[0], feature_scaled_dataset[1])


    NBYF = neural_network(split_data, False) # No batch normalization and feature scaling

    roc_graph(NBYF[2], "Model 2")

    v_loss.append(NBYF[0].history['val_loss'][0])
    t_loss.append(NBYF[0].history['loss'][0])



    """

        This is the model with the best results preprocessing

    """

    eta_v_loss = []
    eta_t_loss = []


    YBYF25 = neural_network(split_data, True, learning_rate=0.25)
    
    eta_v_loss.append(YBYF25[0].history['val_loss'][0])
    eta_t_loss.append(YBYF25[0].history['loss'][0])

    YBYF01 = neural_network(split_data, True, learning_rate=0.01)
    eta_v_loss.append(YBYF01[0].history['val_loss'][0])
    eta_t_loss.append(YBYF01[0].history['loss'][0])

    # ReLU
    YBYF = neural_network(split_data, True) # , batch_training=True) # With batch normalization and feature scaling
    eta_v_loss.append(YBYF[0].history['val_loss'][0])
    eta_t_loss.append(YBYF[0].history['loss'][0])



    """

    We probably wont be using the sigmoid activation function due to the issues it has
    
    """
    # Sigmoid
    sig_YBYF = neural_network(split_data, True, activation_function='sigmoid') # , batch_training=True) # With batch normalization and feature scaling using sigmoid activation function


    v_loss.append(YBYF[0].history['val_loss'][0])
    t_loss.append(YBYF[0].history['loss'][0])



    """
    Note: these functions will be modified so that the graphs are saved as pdf files instead of
    havingt them displayed. 
    """
    # Generate graphs for the best model
    roc_graph(YBYF01[2], "Model 3")
    confusion_matrix(YBYF01[2])
    multi_roc_graph(YBYF01[2], sig_YBYF[2], "ReLU Vs. Sigmoid")
    

    parameter_tuning(v_loss, t_loss, "Different Preprocessing stragaties over the same model")
    parameter_tuning(eta_v_loss, eta_t_loss, "Different learning rate over the same model ReLU")

    batch_YBYF01 = neural_network(split_data, True, learning_rate=0.01, batch_training=True)

    """
    
        Learning curve

    """

    learning_curve(batch_YBYF01[4])

if __name__=="__main__":
    main()