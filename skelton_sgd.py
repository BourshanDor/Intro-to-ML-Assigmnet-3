#################################
# Your name: Dor Bourshan
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.special import expit

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



def SGD_hinge(data:np.ndarray, labels:np.ndarray, C:float, eta_0: float, T: int):
    """
    Implements SGD for hinge loss.
    """
    N = data.shape[0]
    w_t = np.zeros(data.shape[1])

    for t in range(1,T+1) :
        eta_t = eta_0 / t
        i = np.random.randint(0, N)
        x_i = data[i]
        y_i = labels[i]
        if np.dot(x_i, w_t) * y_i < 1 : 
            w_t = np.dot(1-eta_t, w_t)   + np.dot((eta_t * C * y_i) , x_i) 
        else : 
            w_t = np.dot(1-eta_t, w_t)
    
    return w_t

def SGD_log(data: np.ndarray, labels:np.ndarray, eta_0: float, T: int)->tuple:
    """
    Implements SGD for log loss.
    """
    N = data.shape[0]
    w_t = np.zeros(data.shape[1])
    w_t_norm = [] 

    for t in range(1,T+1) :
        w_t_norm.append(LA.norm(w_t))
        eta_t = eta_0 / t
        i = np.random.randint(0, N)
        x_i =  data[i]
        y_i = labels[i]
        w_t = w_t - np.dot(eta_t, gradient_function(y_i, x_i, w_t))
        # w_t = w_t + eta_t * y_i * x_i * expit(-1*y_i * np.dot(w_t, x_i))
     
    return w_t, w_t_norm


#################################

def gradient_function(y:float, x:np.ndarray, w:np.ndarray )-> list : 
    t = -1* y * np.dot(w,x)
    if t > 0 : 
        return np.dot(x,(-1*y) / (1 + np.power(np.e , -1*t)) ) 
    else : 
        p = np.power(np.e , t)
        return np.dot(x,-y*(p / (1 + p)))   



def average_accuracy_plot_eta(question:int , train_data:np.ndarray, train_labels:np.ndarray, validation_data:np.ndarray, validation_labels:np.ndarray, number_of_runs:int,
                                C:float, eta_0_lst:list, T:int, xlim_left:float, xlim_right:float, log: bool) -> float:
     
    y_axis = [] 

    for eta_0 in eta_0_lst:
        accur = 0 
        for i in range(number_of_runs) :
            if question == 1 : 
                w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            else : 
                w = SGD_log(train_data, train_labels, eta_0, T)[0]
            accur += accuracy_linear_classifier(w, validation_data, validation_labels)  
        y_axis.append(accur / number_of_runs)

    accu_of_best_eta_0 = 0
    j = -1 
    for i in range(len(y_axis)): 
        if y_axis[i] > accu_of_best_eta_0 :
            accu_of_best_eta_0 = y_axis[i]
            j = i 
    
    print('***********************************************')
    print('The best eta_0 on avarage is: %.6f' % eta_0_lst[j] )
    print('-----------------------------------------------')
    print('The best accuracy on avarage is: %.3f' % accu_of_best_eta_0)
    print('***********************************************')
    print('-----------------------------------------------')
    print('eta that checked: ' +  str(eta_0_lst[0]) + ', ' + str(eta_0_lst[1]) + ', ..., ' + str(eta_0_lst[len(eta_0_lst)-2]) + ', ' + 
          str(eta_0_lst[len(eta_0_lst)-1]) )
    
    print('***********************************************')   
    plt.plot(eta_0_lst, y_axis) 
    plt.title("averaging the accuracy on the validation set across 10 runs")
    plt.xlabel("eta_0")
    plt.ylabel("averaging the accuracy on the validation set across 10 runs")

    if log : 
      plt.xscale('log')

    plt.xlim(xlim_left, xlim_right)
    plt.legend()
    plt.show()
    return eta_0_lst[j]

def average_accuracy_plot_C(train_data:np.ndarray, train_labels:np.ndarray, validation_data:np.ndarray, validation_labels:np.ndarray,
                             number_of_runs:int, C_lst:list, eta_0:float, T:int, xlim_left:float, xlim_right:float, log:bool) -> float:
    

    y_axis = [] 

    for c in C_lst:
        accur = 0 
        for i in range(number_of_runs) :
            w = SGD_hinge(train_data, train_labels, c, eta_0, T)
            accur += accuracy_linear_classifier(w, validation_data, validation_labels)  
        y_axis.append(accur / number_of_runs)

    accu_of_best_C = 0
    j = -1 
    for i in range(len(y_axis)): 
        if y_axis[i] > accu_of_best_C :
            accu_of_best_C = y_axis[i]
            j = i 
    
    print('***********************************************')
    print('The best C on avarage is: %.6f' % C_lst[j] )
    print('-----------------------------------------------')
    print('The best accuracy on avarage is: %.3f' % accu_of_best_C)
    print('***********************************************')
    print('-------------------------------------------')
    print('C that checked: ' +  str(C_lst[0]) + ', ' + str(C_lst[1]) + ', ..., ' + str(C_lst[len(C_lst)-2]) + ', ' + 
          str(C_lst[len(C_lst)-1]) )
    print('***********************************************')

    plt.plot(C_lst, y_axis) 
    plt.title("averaging the accuracy on the validation set across 10 runs")
    plt.xlabel("C")
    plt.ylabel("averaging the accuracy on the validation set across 10 runs")

    if log : 
      plt.xscale('log')

    plt.xlim(xlim_left, xlim_right)
    plt.legend()
    plt.show()
    return C_lst[j]


def accuracy_linear_classifier(w :np.ndarray , validation_data: np.ndarray, validation_labels:np.ndarray) : 
    linear_classifier = lambda x : np.sign(np.dot(w, x))
    correct_classify = lambda y_hat, y : y_hat == y

    faild_number = 0 
    N = len(validation_data)

    for i in range(N)  :
        if not correct_classify(linear_classifier(validation_data[i]), validation_labels[i]):
            faild_number += 1 

    return 1 - (faild_number / N)

def w_as_picture(question, train_data, train_labels, test_data, test_labels,best_eta, best_C, T):
    if question == 1 : 
        w = SGD_hinge(train_data, train_labels, best_C, best_eta, T)
    else : 
        w= SGD_log(train_data, train_labels, best_eta, T)[0]

    accur = accuracy_linear_classifier(w, test_data, test_labels)
    print('***********************************************')
    print('The accuracy of the best classifier: %.6f' % accur )
    print('***********************************************')

    plt.imshow(np.reshape(w,(28,28)),interpolation = 'nearest' )
    plt.show()

def w_norm_change(train_data, train_labels, eta, T) : 
    w, w_norm = SGD_log(train_data, train_labels, eta, T)
    x_axis = range(1,T+1) 
    plt.plot(x_axis, w_norm) 
    plt.title("norm change as SGD progresses")
    plt.xlabel("t")
    plt.ylabel("||w_t||")
    plt.show()


def main() : 
    
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    # eta_0_lst = [10**i for i in range(-5,6,1)]
    # best_eta = average_accuracy_plot_eta(1,train_data, train_labels, validation_data, validation_labels, 10, 1, eta_0_lst, 1000, 0, 10**5, True)

    # eta_0_lst = [i for i in np.arange(-10000 + best_eta, best_eta + 10000,100)]
    # best_eta = average_accuracy_plot_eta(1,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -10000 + best_eta, best_eta + 10000, False)

    # eta_0_lst = [i for i in np.arange(-500 + best_eta, best_eta + 500,10)]
    # best_eta = average_accuracy_plot_eta(1,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -500 + best_eta, best_eta + 500, False)

    # eta_0_lst = [i for i in np.arange(-50 + best_eta, best_eta + 50,1)]
    # best_eta = average_accuracy_plot_eta(1,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -50 + best_eta, best_eta + 50, False)

    # eta_0_lst = [i for i in np.arange(-10 + best_eta, best_eta + 10, 0.1)]
    # best_eta = average_accuracy_plot_eta(1,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -10 + best_eta, best_eta + 10, False)

    # C_lst = [10**(i) for i in range (-5,6,1)]
    # best_c = average_accuracy_plot_C(train_data, train_labels, validation_data, validation_labels,10, C_lst, best_eta , 1000, 0, 10**5, True)

    # C_lst = [i for i in np.arange(-10000 + best_c,best_c + 10000, 100)]
    # best_c = average_accuracy_plot_C(train_data, train_labels, validation_data, validation_labels,10, C_lst, best_eta , 1000, -10000 + best_c, best_c + 10000, False)

    # C_lst = [i for i in np.arange(best_c - 500 ,best_c + 500, 10 )]
    # best_c = average_accuracy_plot_C(train_data, train_labels, validation_data, validation_labels,10, C_lst, best_eta , 1000,best_c - 500 ,best_c + 500, False)

    # C_lst = [i for i in np.arange(best_c - 50 ,best_c + 50, 1 )]
    # best_c = average_accuracy_plot_C(train_data, train_labels, validation_data, validation_labels,10, C_lst, best_eta , 1000,best_c - 50 ,best_c + 50, False)

    # C_lst = [i for i in np.arange(best_c - 10 ,best_c + 10, 0.1 )]
    # best_c = average_accuracy_plot_C(train_data, train_labels, validation_data, validation_labels,10, C_lst, best_eta , 1000,best_c - 10 ,best_c + 10, False)


    # w_as_picture(1,train_data, train_labels, test_data, test_labels, best_eta, best_c, 20000 )

    eta_0_lst = [10**i for i in range(-5,6,1)]
    best_eta = average_accuracy_plot_eta(2,train_data, train_labels, validation_data, validation_labels, 10, 1, eta_0_lst, 1000, 0, eta_0_lst, True)

    # eta_0_lst = [i for i in np.arange(-10000 + best_eta, best_eta + 10000,100)]
    # best_eta = average_accuracy_plot_eta(2,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -10000 + best_eta, best_eta + 10000, False)

    # eta_0_lst = [i for i in np.arange(-500 + best_eta, best_eta + 500,10)]
    # best_eta = average_accuracy_plot_eta(2,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -500 + best_eta, best_eta + 500, False)

    # eta_0_lst = [i for i in np.arange(-50 + best_eta, best_eta + 50,1)]
    # best_eta = average_accuracy_plot_eta(2,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -50 + best_eta, best_eta + 50, False)

    # eta_0_lst = [i for i in np.arange(-10 + best_eta, best_eta + 10, 0.1)]
    # best_eta = average_accuracy_plot_eta(2,train_data, train_labels, validation_data, validation_labels,10, 1, eta_0_lst, 1000, -10 + best_eta, best_eta + 10, False)

    w_as_picture(2,train_data, train_labels, test_data, test_labels, best_eta, 0, 20000 )
    w_norm_change(train_data, train_labels, 10**(-5), 20000)
    

if __name__ == "__main__":
    main()
    