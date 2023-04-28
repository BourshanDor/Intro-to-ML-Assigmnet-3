#################################
# Your name: Dor Bourshan
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    # TODO: Implement me

    N = data.size 
    w_t = 0
    eta_update = lambda t : eta_0 / t 
    eta_t = eta_0

    for t in range(T) :
        eta_t = eta_update(t)
        i = np.random.randint(0, N)
        x_i = data[i]
        y_i = labels[i]
        if np.multiply(x_i, w_t) * y_i < 1 : 
            w_t = (1-eta_t) * w_t + eta_t * C * y_i * x_i 
        else : 
            w_t = (1-eta_t) * w_t
    
    return w_t




    pass


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    pass

#################################

 def cross_validation(train_data, train_labels, validation_data, validation_labels, C , T):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.
        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        
        w = SGD_hinge(train_data, train_labels,C, eta_0, T)
        training_number = int(m*0.8) 
        test_number = m - training_number

        x_axis = [] 
        y_true_axis = [] 
        y_empirical_axis = []

        k_of_the_best_empirical_error =  0
        best_empirical_error = 1 
        best_hypothesis = []
        train_sample = self.sample_from_D(training_number)
        test_sample = self.sample_from_D(test_number)
        train_sample_x = train_sample[:,0] 
        train_sample_lable = train_sample[:,1] 
    
        for k in range(1, 11):
            interval, besterror = intervals.find_best_interval(train_sample_x, train_sample_lable, k)
            y_true_axis.append(self.true_error(interval))
            empirical_error = self.empirical_error(interval, test_sample)
            y_empirical_axis.append(empirical_error) 
            x_axis.append(k)

            if best_empirical_error > empirical_error : 
                k_of_the_best_empirical_error = k
                best_empirical_error = empirical_error
                best_hypothesis = interval

        print("The best empirical k : %d" % k_of_the_best_empirical_error)
        print("The best hyposesis that found represent as a set of intervals: ", best_hypothesis)

            
        plt.plot(x_axis, y_true_axis, label="True Error")
        plt.plot(x_axis, y_empirical_axis, label="Empirical Error")
        plt.title("empirical and true errors as a function of k")
        plt.xlabel("k")
        plt.ylabel("empirical and true errors ")
        plt.legend()
        plt.show()


#################################