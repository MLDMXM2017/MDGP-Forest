# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 21:13:59 2024

@author: 12207
"""
import math
import numpy as np
import copy
from collections import Counter

from .GPFeatureConstructor import GPFeatureConstructor

from concurrent.futures import ProcessPoolExecutor

def createOVADataset(x, y, y_probs, hardness_threshold, confidences):
    unique_classes = np.unique(y)
    ova_x = []
    ova_y = []
    for cls in unique_classes:  
        y_ova = (y == cls).astype(int)
        probabilities = y_probs[np.arange(x.shape[0]), cls]
        x_ova = x.copy()
        
        count_0 = np.sum(y == 0)  
        count_1 = np.sum(y == 1)
        if count_0 > count_1:  
            remove_indices = (y == 0) & ((confidences > hardness_threshold) | (probabilities > hardness_threshold))
        else:
            remove_indices = (y == 1) & (confidences > hardness_threshold)
        x_ova = x_ova[~remove_indices]  
        y_ova = y_ova[~remove_indices] 

        ova_y.append(y_ova)
        ova_x.append(x_ova)
         
    return ova_x, ova_y

def wrapper_fitSinglePop(args):
    return fitSinglePop(*args)

def fitSinglePop(input_num, pop_num, features_num, x, y, enhance_vector_len, generation, cxProb, mutProb):
    print("Start training GP.")
    
    enhance_vector = None
    if enhance_vector_len > 0:
        enhance_vector = x[:, -enhance_vector_len:]
        x = x[:, :-enhance_vector_len]
        
    gpfc = GPFeatureConstructor(input_num, pop_num, features_num)
    gpfc.fit(x, y, enhance_vector, generation, cxProb, mutProb)
    print("GP training completed.")
    return gpfc

class CategoryRelatedFC():
    def __init__(self, input_num, categories_num):
        self.input_num = input_num
        self.categories_num = categories_num
        self.gpfcs = {}
        
    def copyGPFC(self, pre_layer, new_layer):
        self.gpfcs[new_layer] = copy.deepcopy(self.gpfcs[pre_layer])
    
    def fit(self, layer, pop_num, features_num, x, y, y_probs=None, enhance_vector_len = 0, hardness_threshold = 0.95, generation = 20, cxProb = 0.5, mutProb= 0.2):
        
        print("train layer " + str(layer) + " feature")
        confidences = y_probs[np.arange(x.shape[0]), y]
        ova_x, ova_y = createOVADataset(x, y, y_probs, hardness_threshold, confidences)
        
        params = [(self.input_num, pop_num, features_num, ova_x[i], ova_y[i], enhance_vector_len, generation, cxProb, mutProb) for i in range(self.categories_num)]
        with ProcessPoolExecutor(max_workers=self.categories_num) as pool:
            return_results = list(pool.map(wrapper_fitSinglePop, params))
            
        self.gpfcs[layer] = return_results
        print()
        
    def transform(self, x, layer, enhance_vector_len):
        enhance_vector = None
        if enhance_vector_len > 0:
            enhance_vector = x[:, -enhance_vector_len:]
            x = x[:, :-enhance_vector_len]
            
        new_features = []
        for gpfc in self.gpfcs[layer]:
            new_features.append(gpfc.transform(x))
        new_x = np.concatenate((new_features), axis=1)
        if enhance_vector_len > 0:
            new_x = np.concatenate((new_x, enhance_vector), axis=1)
        return new_x