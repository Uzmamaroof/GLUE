#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import stumpy
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

#plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')


# In[3]:


#steam_df = pd.read_csv("https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1")

raw_data = np.load("../NoDef.npz")
raw_labels = raw_data['labels']
raw_traces = raw_data['traces']


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
    
    starmap = []
    
    # generate list of tuple arguments to be passed to pool
    for i in range(sample_size):
        s1 = s1_sample[i].astype('float64')
        for j in range(i, sample_size):
            s2 = s2_sample[j].astype('float64')
            length = min(subseq_len, len(s1), len(s2))
            #starmap.append(("s1_" + str(i),"s2_" + str(j),length))
            starmap.append((s1,s2,length))
    
    #print(starmap)
    #print("Comparisons to make: " + str(len(starmap)))
    
    
    with Pool(num_threads) as p:
        result = p.starmap(stumpy.gpu_mpdist, starmap)
        #print(result)
    
    
    return statistics.mean(result)


# In[14]:


# TRAINING MODE
# SECOND VERSION - MULTI THREAD

final_scores = np.zeros((100,100))




for i in tqdm(range(100)):
    for j in tqdm(range(i,100)):
        final_scores[i][j] = calculate_scores_threaded(traces[i],traces[j],sample_size=2,subseq_len=250)


# In[230]:


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

