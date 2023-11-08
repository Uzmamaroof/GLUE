#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from multiprocessing import Pool

# PART 2
if __name__ == '__main__':
    
    global traces

    with open('../nonzero_traces.npy', 'rb') as f:
        traces = pickle.load(f)
    
    shapelet_samples_list = [4, 5, 6]
    shapelet_size_list = [300, 400, 500, 600]
    clf_samples_list = [400000]

    from utils import evaluate_parameters
    
    parameter_list = np.array(np.meshgrid(shapelet_samples_list, shapelet_size_list, clf_samples_list)).T.reshape(-1,3)
    
    for parameters in parameter_list:
        evaluate_parameters(parameters[0], parameters[1], parameters[2])