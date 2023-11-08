import stumpy
import numpy as np
import datetime as dt
import random
import math
import pickle
import sys

from statistics import mean
from tqdm.auto import tqdm
from multiprocessing import Pool

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


'''
Collects random samples from trace with id2 and computes the matrix profile of class1 compared with class 2

Input: 
    trace1: packet traces from class 1
    id2: id number for class 2 
    num_traces: number of traces to select from class 2 (should be equal to class 1)
    shapelet_size: length of shapelets
    
Output:
    Matrix profile of trace1 compared with trace2
'''
def compare_profile(trace1, id2, num_traces, shapelet_size):
    
    trace2 = []
    while len(trace2) < shapelet_size:
        trace2 = random.sample(traces[id2], num_traces)
        trace2 = np.asarray([item for row in trace2 for item in row]).astype('float64')
    
    #print("self-profiles generated...")
    c1_c2 = stumpy.stump(trace1, shapelet_size, trace2, ignore_trivial=False)[:, 0].astype(float)
    c1_c2[c1_c2 == np.inf] = np.nan
    #print("Comparison profiles generated...")
    
    return c1_c2

'''
Compares a the matrix profile of a class trace with itself

Input: 
    trace: packet traces from class 1
    shapelet_size: length of shapelets
    
Output:
    Matrix profile of trace compared with trace
'''

def same_profile(trace, shapelet_size):
    
    c1_c1 = stumpy.stump(trace, shapelet_size)[:, 0].astype(float)
    c1_c1[c1_c1 == np.inf] = np.nan
    
    return c1_c1

'''
return indices of shapelet as one-hot encoded list
'''
def generate_shapelet(trace, diff, shapelet_size):
    
    idx = np.argmax(diff)
    shapelet = np.asarray([1 if idx <= i < idx + shapelet_size else 0 for i in range(len(trace))])
    
    return shapelet

'''
Compute shapelet of greatest overlaps
'''
def find_overlap(trace_i, shapelets_i, shapelet_size):
    #print(shapelets_i[0])
    
    merged_shapelets = np.sum(shapelets_i, axis=0)
    
    max_size = 0
    start = 0
    end = 0
    
    for i in range(0, len(merged_shapelets), shapelet_size):
        current_size = np.sum(merged_shapelets[i:i+shapelet_size])
        if current_size > max_size:
            max_size = current_size
            start = i
            end = i + shapelet_size
    
    return trace_i[start:end]


# In[ ]:


'''
Generates a set of 100 shapelets for each class in samples

Input:
    num_traces = Number of traces per class
    shapelet_size = Size of shapelets
    save: save results to file?
    filename: if save, name & location of output file

Output:
    list object containing shapelets for each class

'''

def generate_shapelets(num_traces, shapelet_size, save=True, filename="../shapelets.pkl"):
    shapelet_storage = []
    
    # loop over all classes (generate shapelet for each class)
    for i in tqdm(range(100)):
        
        # get num_traces samples from trace #i
        # while loop guarantees that the traces selected exceed shapelet size (or crashes will happen)
        trace_i = []
        while len(trace_i) < shapelet_size:
            trace_i = random.sample(traces[i], num_traces)
            trace_i = np.asarray([item for row in trace_i for item in row]).astype('float64')
        
        shapelets_i = np.zeros((100, len(trace_i)))
        #print(shapelets_i.shape)
        
        # generate profile of i compared with itself
        ci_ci = same_profile(trace_i, shapelet_size)
        
        # loop over every other class and generate a profile for each one
        for j in range(100):
            # don't compare i with itself 
            if i == j:
                continue

            # compute profile of i compared with j
            ci_cj = compare_profile(trace_i, j, num_traces, shapelet_size)

            # find largest value gap between other and i
            diff_ci = ci_cj - ci_ci
            
            # generate best shapelet for i compared to j and store it in list
            ci_shape = generate_shapelet(trace_i, diff_ci, shapelet_size)
            shapelets_i[j] = ci_shape
        
        # compare shapelets between all classes and return the one which has the most overlap
        # (i.e.) the shapelet that was chosen most between the 99 other classes
        best_shapelet = find_overlap(trace_i, shapelets_i, shapelet_size)
        # save to list
        shapelet_storage.append(best_shapelet)
    
    # save list to file
    if save:
        with open(filename, 'wb') as f:
            pickle.dump(shapelet_storage, f)
    
    return shapelet_storage   


# In[ ]:


'''
Compute the minimum distance beteen data samples and shapelets
Input:
    data = list of individual packet traces
    shapelets = list of shapelets
Output:
    minimum distance between each sample in data compared with each sample in shapelet
    shape = (len(data),len(shapelets))
'''
def distance_to_shapelet(data, shapelets):
    #data = np.asarray(data)
    #print(len(data))
    
    # processed output data
    data_out = np.zeros((len(data),len(shapelets)))
    
    # loop over each sample in the dataset
    for i,sample in enumerate(tqdm(data)):
        shapelet_score = np.empty(len(shapelets))
        # for each shapelet, calculate distance and assign a score
        for j,shapelet in enumerate(shapelets):
            dist = stumpy.mass(shapelet, sample)
            shapelet_score[j] = dist.min()
        data_out[i] = shapelet_score
    
    return data_out

'''
Computes distances between input samples and shapelets, returns X for classifier
Also cleans data and ensures no random errors due to length, NaN, etc...
Underlying function that performs comparison is distance_to_shapelet
Selects data samples (with replacement)
note: some samples will always be bad so actual length of X is less

Input:
    num_traces = numner of traces to process
    save = save output to file
    filenames = tuple that represents (name of X file, name of y file)

Output:
    X values for classifier of shape (None, 100)
    y values for classifier of shape (None, )
'''

def process_traces(num_traces, shapelets, shapelet_size, save=True, filenames=("X.pkl","y.pkl")):
    X, y = [], []

    for i in range(num_traces):
        random_id = random.randrange(100)
        random_trace = random.choice(traces[random_id])
        X.append([random_trace])
        y.append(random_id)

    # process and remove useless entries (too short)
    X = [np.asarray(trace).astype('float64') for trace in X]
    X = [trace[~np.isnan(trace)] for trace in X]    
    removals = [i for i,x in enumerate(X) if len(x) < shapelet_size]
    for idx in removals:
        X[idx] = None
        y[idx] = None
    X = [trace for trace in X if trace is not None]
    y = [value for value in y if value is not None]

    # compute distance between input trace and shapelet arrays
    # return as new X

    X = distance_to_shapelet(X, shapelets)
    
    if save:
        with open(filenames[0], 'wb') as f:
            pickle.dump(X, f)

        with open(filenames[1], 'wb') as f:
            pickle.dump(y, f)
    
    return X, y


# In[ ]:


'''
Evaluate performance of sklearn classifier on data samples - 90/10 training testing split

Input:
    clf: sklearn classifier object
    X: x values
    y: y values
    topk: k values for evaluation metrics
Output:
    list of length topk with accuracy for testing data
'''

def classifier_performance(clf, X, y, topk=[1,3,5]):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    
    scores = []
    for k in topk:
        correct = 0
        for i in range(len(y_prob)):
            ind = np.argpartition(y_prob[i], -k)[-k:]
            if y_test[i] in ind:
                correct += 1
        scores.append(correct/len(y_prob))
    
    return scores


# In[ ]:


'''
Utility function for pipeline of evaluating different grid search parameters
Output: a new file located in ../results/param1-val1_param2-val2_param3-val3
        the file contains a pickled python object
        with the scores for top-1, top-3, and top-5 classifier accuracy
'''
# note: python multiprocessing is really annoying to work with
# function needs to be in a separate .py file which is imported
# and function can only have 1 argument
# list input which is immediately used for what would be the arguments
def evaluate_parameters(arr):
    
    global traces
    with open('../nonzero_traces.npy', 'rb') as f:
        traces = pickle.load(f)
    
    num_shapelet_samples = arr[0]
    shapelet_size = arr[1]
    num_clf_samples = arr[2]
    
    print("Generating Shapelets ...")
    shapelets = generate_shapelets(num_shapelet_samples, shapelet_size, False)
    
    print("Processing Traces...")
    X, y = process_traces(num_clf_samples, shapelets, shapelet_size, False)
    
    print("Evaluating Classifier...")
    clf = RandomForestClassifier()
    scores = classifier_performance(clf, X, y)
    
    outfile_name = "../results/spts-" + str(num_shapelet_samples) + "_size-" + str(shapelet_size) + "_Xs-" + str(num_clf_samples)
    with open(outfile_name, 'wb') as f:
        pickle.dump(scores, f)


# In[ ]:


# shapelet_samples_list = [3, 4, 5]
# shapelet_size_list = [300, 400, 500, 600]
# clf_samples_list = [400000, 500000, 600000]


shapelet_samples_list = [3, 4, 5]
shapelet_size_list = [300, 400, 500, 600]
clf_samples_list = [1000, 2000, 3000]

grid_parameters = np.array(np.meshgrid(shapelet_samples_list, shapelet_size_list, clf_samples_list)).T.reshape(-1,3)

print(grid_parameters.shape)

evaluate_parameters([3,600,2000])