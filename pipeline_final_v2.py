#!/usr/bin/env python
# coding: utf-8

# In[1]:

import stumpy
from stumpy import core
from stumpy.gpu_stump import gpu_stump
from stumpy.gpu_aampdist import gpu_aampdist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt
import random
import math
import statistics
from tqdm.auto import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat
import functools

#print("imports done")
#plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')


# In[2]:


# from cuda import cuda
# try:
#    mp.set_start_method('spawn', force=True)
# except RuntimeError:
#    pass


# In[3]:


#steam_df = pd.read_csv("https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1")

raw_data = np.load("../NoDef.npz")
raw_labels = raw_data['labels']
raw_traces = raw_data['traces']

print("data loaded")


# In[4]:


trace_ids = list(set(raw_labels))


# In[5]:


'''
Convert a raw packet trace into a processed version
trace: 1-D numpy array containing a packet trace
mode:
    'f': flatten the trace using packet arrival time
    'p': only include positive
    'n': only include negative
    'z': only remove zeroes
granularity: in 'f' mode, the smallest distance between packet arrival times to be considered
remove_zeroes: remove any zero values from the packet trace before processing (excluding z mode)
maxlen: reshape input list to be this length after processing by padding with 0
'''
def process(trace, mode='f', granularity=0.01, remove_zeroes=True, maxlen=10000):
    if remove_zeroes:
        trace = trace[trace != 0]
    
    if mode == 'f':
        processed_trace = []
        for i,x in enumerate(trace):
            distance = abs(x) - abs(trace[i-1])
            num_zeroes = math.ceil(distance/granularity)
            processed_trace += [0] * num_zeroes
            processed_trace.append(np.sign(x))
        return processed_trace + [0.0] * (maxlen - len(processed_trace))
    elif mode == 'p':
        return trace[trace >= 0]
    elif mode == 'n':
        return trace[trace <= 0]
    elif mode == 'z':
        return trace[trace != 0]
    else:
        raise TypeError("mode must be one of: f,p,n,z")


# In[6]:


traces = {id: [None] * 4500 for id in trace_ids}
indices = {id: 0 for id in trace_ids}


# In[7]:


#print(raw_traces[345][0:20])
#print(len(raw_traces[345]))
#test = process(raw_traces[345], mode='f', granularity=0.01, remove_zeroes=True)
#print(len(test))
#print(test[9950:])


# In[8]:


# dictionary of all the traces as 2D numpy arrays
# keeping track of index for each one and setting value is necessary
# simple append is way too slow (>30 mins runtime)

for i in tqdm(range(len(raw_traces))):
    processed_trace = process(raw_traces[i], mode='z', remove_zeroes=True)
    
    traces[raw_labels[i]][indices[raw_labels[i]]] = processed_trace
    indices[raw_labels[i]] += 1


# In[9]:


print(len(traces))
print(len(traces[0]))
# print(len(traces[45]))
print(len(traces[38][8]))
# print(len(traces[45][50]))


# In[10]:


# def get_sample_index(trace_list, mode):
#     for i in range(len(trace_list)):
#         if len(trace_list[i]) == mode:
#             return i

def generate_primary_sample(trace_list, mode='mode_avg'):
    
    if mode == "mode_avg":
        # get the mode trace length
        mode = statistics.mode([len(trace) for trace in primary_trace])
        # get a list of all traces of mode length
        mode_traces = np.asarray([trace for trace in primary_trace if len(trace) == mode])
        # return vertical average of traces
        return np.mean(mode_traces, axis=0)
    elif mode == 'mode_single':
        mode = statistics.mode([len(trace) for trace in primary_trace])
        for trace in trace_list:
            if len(trace) == mode:
                return trace


# In[13]:


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

'''
function for computing the complete stumpy scores with a multi-threaded approach

s1: the primary trace for which scores are being computed
s2: the other trace being compared with the primary
sample_size: how many traces should be randomly selected from each trace set (default: 5)
subseq_len: length of subsequences to be compared (default: 250)

Requirements:
s1, s2 are length 100
sample_size is divisible by 100

Returns:
final score for the average distance between s1 and s2


'''
            
def calculate_scores_threaded(
    s1_list, s2_list, 
    sample_size = 2, 
    subseq_len = 250,
    num_threads = 25):
    
    #print(sample_size)
    #print(subseq_len)
    s1_sample = random.sample(s1_list,sample_size)
    s2_sample = random.sample(s2_list,sample_size)
    
    partial_gpu_stump = functools.partial(
        gpu_stump,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )
    
    starmap = []
    
    # generate list of tuple arguments to be passed to pool
    for i in range(sample_size):
        s1 = s1_sample[i].astype('float64')
        for j in range(i, sample_size):
            s2 = s2_sample[j].astype('float64')
            length = min(subseq_len, len(s1), len(s2))
            
            starmap.append((s1,s2,length,0.05,partial_gpu_stump,0.05,None,None,None,None))
    
#     print(starmap)
#     print("Comparisons to make: " + str(len(starmap)))
    
#     args_iter = zip(repeat(project_name), api_extensions)
#     kwargs_iter = repeat(dict(payload={'a': 1}, key=True))
#     branches = starmap_with_kwargs(pool, fetch_api, args_iter, kwargs_iter)
    
    with Pool(num_threads) as p:
        result = p.starmap(stumpy.core._mpdist, starmap)
    
    #all_gpu_devices = [device.id for device in cuda.list_devices()]
    
    #print(all_gpu_devices)
    #stumpy.mpdist(s1,s2,length, device=all_gpu_devices)
    
    return statistics.mean(result)


# In[18]:


# TRAINING MODE
# SECOND VERSION - MULTI THREAD

final_scores = np.zeros((100,100))

for i in tqdm(range(1)):
    for j in tqdm(range(i,1)):
        final_scores[i][j] = calculate_scores_threaded(traces[i],traces[j],sample_size=2,subseq_len=250, num_threads=128)
    
             


# In[ ]:


correct_count = 0
incorrect_count = 0

i=0

for i in range(100):
    base_score = final_scores[i][i]

    sample_i_correct = 0
    for j in range(i, 100):
        if final_scores[i][j] > base_score:
            correct_count += 1
            sample_i_correct += 1
        else:
            incorrect_count += 1
    
    #print("Sample " + str(i) + " correct:" + str(sample_i_correct))
        
print("Correct: " + str(correct_count))
print("Incorrect: " + str(incorrect_count))

print("Correct (%): " + str(correct_count/(incorrect_count + correct_count) * 100))

