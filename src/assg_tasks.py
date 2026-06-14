import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import scipy


# you need create class EveryNEpochs here to subclass keras.callbacks.Callback
# don't forget PyDoc documenation for functions, and also member methods
            
# you need to implement get_unregularized_model() here.  Don't forget PyDoc documentation
# for your function

# you need to implement get_l2_regularized_model() here.  Don't forget PyDoc documentation
# for your function

# you need to implement get_dropout_model() here.  Don't forget PyDoc documentation
# for your function


def load_soccer_dataset():
    """Load the training and test data for the soccer dataset for
    this assignment.  The data is is a matlab .mat file, so use
    scipy io function to read it. 

    Returns
    -------
    train_X, train_y : ndarray(211,2), ndarray(211,)
      The training data, there are m=211 samples in this data, the input features are the x and y
      coordinates on the soccer field of the head intercept position
    test_X, test_y  : ndarray(200,2), ndarray(200,) 
      corresponding test data, there are m=200 samples for testing
    """
    # load the data
    data = scipy.io.loadmat('../data/soccer_data.mat')
    train_X = data['X']
    train_y = data['y'].flatten()
    test_X = data['Xval']
    test_y = data['yval'].flatten()

     # randomly shuffle data since it is currently sorted by category
    permutation = np.random.permutation(len(train_y))
    train_X = train_X[permutation]
    train_y = train_y[permutation]
    
    return train_X, train_y, test_X, test_y

def plot_history(ax, history_dict, metric_key):
    """Plot the asked for metrics. Usually we need to plot the metric from the training
    data and its corresponding measurement using the validation data, thus we pass in
    two keys for the training and validation metric to plot.

    Arguments
    ---------
    ax : matplotlib.axis
      a matplotlib figure axis to create plot onto
    history_dict : dict 
      A Python dictionary whose keys should return list like enumerable
      items holding the measured metrics over some number of epochs of training.
    metric_key : str
      The string key for the metric, validation data is assumed to be
      accessible as "val_" + metric_key


    """
    # setup epochs and keys/labels for the plot
    train_key = metric_key
    train_label = "Training " + metric_key
    val_key = "val_" + metric_key
    val_label = "Validation " + metric_key
    epochs = np.arange(1, len(history_dict[train_key]) + 1)
    
    # create the plot of the train and test metric
    ax.plot(epochs, history_dict[train_key], 'r-', label=train_label)
    ax.plot(epochs, history_dict[val_key], 'b-', label=val_label)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_key)
    #ax.set_xticks(epochs)
    ax.grid()
    ax.legend();

class KerasClassifierWrapper:
    """Apparently mlxtend requires that the classifier return integer predictions, not
    floats.  Keras trained models return a float between 0.0 or 1.0 when predict is called
    (e.g. more like predict probability), instead of thresholding the class to either 0 or 1
    for a binary classification like we are doing here.  So need to wrap the Keras classifier
    model predict function to return what is expected by the mlxtend plot_decision_regions()
    function.

    TODO: probably should move this into utilities for assignments and add in description of
    using this for the mlxtend plot_decision_regions() function.
    """
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        """Wrap the predict function to threshoold classification results to
        integer 0.0 or 1.0 results as expected by the mlxtend decision region plot
        functions.

        Parameters
        ----------
        self - This is a member method of the wrapper class, so pass in reference to self
          as first parameter
        X - The data to make predictions for.
        """
        # get the original predictions that are floats from 0.0 to 1.0 from the original model
        y_pred = self.model.predict(X, verbose=0)

        # threshold and reuturn integer 0/1 binary class results
        return (y_pred > 0.5).astype(np.int32).ravel()