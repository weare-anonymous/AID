"""
This script contains functions for generating synthetic data. 

Part of the code is based on https://github.com/Jianbo-Lab/CCM
""" 
from __future__ import print_function
import numpy as np  
from scipy.stats import chi2
from sklearn.model_selection import train_test_split


def generate_XOR_labels(X):
    y = np.exp(X[:,0]*X[:,1])
    prob_1 = np.expand_dims(1 / (1+y) ,1)
    return prob_1

def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 
    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    return prob_1

def generate_additive_labels(X):
    logit = np.exp(-100 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 
    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    return prob_1

def generate_data(datatype, n=10000, seed = 0):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples 
        seed: random seed used
    Return: 
        X(float): [n,d].  
        y(float): n dimensional array. 
    """

    np.random.seed(seed)
    X = np.random.randn(n, 10)
    if datatype == 'XOR':
        y = np.array(generate_XOR_labels(X) > 0.5, dtype=np.float32)
    elif datatype=='orange_skin':
        y = np.array(generate_orange_labels(X) > 0.5, dtype=np.float32)
    elif datatype == 'nonlinear_additive':  
        y = np.array(generate_additive_labels(X) > 0.5, dtype=np.float32)
    elif datatype == 'fusion_feature':
        y1 = np.array(generate_XOR_labels(X) > 0.5, dtype=np.float32)
        y2 = np.array(generate_orange_labels(X) > 0.5, dtype=np.float32)
        y3 = np.array(generate_additive_labels(X) > 0.5, dtype=np.float32)
        y = np.concatenate([y1,y2,y3], axis = 1) 
        valid_data = (np.sum(y, axis=1) == 1)
        final_X = X[valid_data]
        final_y = y[valid_data]
        while final_y.shape[0] < n:
            seed += 1
            np.random.seed(seed)
            X = np.random.randn(n, 10)
            y1 = np.array(generate_XOR_labels(X) > 0.5, dtype=np.float32)
            y2 = np.array(generate_orange_labels(X) > 0.5, dtype=np.float32)
            y3 = np.array(generate_additive_labels(X) > 0.5, dtype=np.float32)
            y = np.concatenate([y1,y2,y3], axis = 1) 
            valid_data = (np.sum(y, axis=1) == 1)
            final_X = np.concatenate([final_X, X[valid_data]])
            final_y = np.concatenate([final_y, y[valid_data]])
        X = final_X[:n]
        y = final_y[:n]
    elif datatype == 'fusion_feature_new':
        y1 = np.array(generate_XOR_labels(X) > 0.5, dtype=np.float32)
        y3 = np.array(generate_additive_labels(X) > 0.5, dtype=np.float32)
        y = np.concatenate([y1,y3], axis = 1) 
        valid_data = (np.sum(y, axis=1) == 1)
        final_X = X[valid_data]
        final_y = y[valid_data]
        while final_y.shape[0] < n:
            seed += 1
            np.random.seed(seed)
            X = np.random.randn(n, 10)
            y1 = np.array(generate_XOR_labels(X) > 0.5, dtype=np.float32)
            y3 = np.array(generate_additive_labels(X) > 0.5, dtype=np.float32)
            y = np.concatenate([y1,y3], axis = 1) 
            valid_data = (np.sum(y, axis=1) == 1)
            final_X = np.concatenate([final_X, X[valid_data]])
            final_y = np.concatenate([final_y, y[valid_data]])
        X = final_X[:n]
        y = final_y[:n]
        
    elif datatype == 'multitask':  
        y1 = np.array(generate_XOR_labels(X) > 0.5, dtype=np.float32)
        y2 = np.array(generate_orange_labels(X) > 0.5, dtype=np.float32)
        y3 = np.array(generate_additive_labels(X) > 0.5, dtype=np.float32)
        y = np.concatenate([y1,y2,y3], axis = 1) 

    # Permute the instances randomly.
    perm_inds = np.random.permutation(n)
    X, y = X[perm_inds], y[perm_inds]
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)

    return x_train, x_val, y_train, y_val

if __name__ == '__main__':

    x_train, x_val, y_train, y_val = generate_data(datatype='fusion_feature_new', n=10000)


