# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:50:35 2024

@author: 12207
"""
import random
import numpy as np

from deap import base, creator, tools, gp
from .algorithms import eaSimple

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler
    
def generate_random_minus_one_to_one():  
    return random.random() * 2 - 1  

def analytical_quotient(x1, x2):
    return x1 / np.sqrt(1 + (x2 ** 2))

def protect_sqrt(a):
    return np.sqrt(np.abs(a))

def evaluate(individuals, pset, x, y, enhance_vector):
    
    new_features = []
    for ind_num, ind in enumerate(individuals):
        func = gp.compile(expr=ind, pset=pset)
        new_features.append([func(*record) for record in x])
    
    new_x = np.transpose(np.array(new_features))
    scaler = MinMaxScaler()
    new_x = scaler.fit_transform(new_x)

    if enhance_vector is not None:
        new_x = np.concatenate((new_x, enhance_vector), axis=1)
        
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(new_x, y)
    importances = clf.feature_importances_
    
    if enhance_vector is not None:
        importances = importances[:-enhance_vector.shape[1]]

    return importances

class GPFeatureConstructor:
    def __init__(self, input_num, pop_num, features_num):
        self.input_num = input_num
        self.features_num = features_num
        self.pop_num = pop_num
        
        self.normalizer = None
        
        self.log = None
        self.hof = None
        
        pset = gp.PrimitiveSet("MAIN", input_num)
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(analytical_quotient, 2)
        pset.addPrimitive(protect_sqrt, 1)
        #pset.addPrimitive(np.sin, 1)
        #pset.addPrimitive(np.cos, 1)
        pset.addPrimitive(np.negative, 1)
        
        pset.addEphemeralConstant("rand101", generate_random_minus_one_to_one)
        self.pset = pset
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.register("compile", gp.compile, pset=pset)
        
        toolbox.register("evaluate", evaluate)
        
        self.toolbox = toolbox
        
        self.pop = self.toolbox.population(n = pop_num)
        self.hof = tools.HallOfFame(features_num)
        
    def fit(self, x, y, enhance_vector = None, generation = 20, cxProb = 0.5, mutProb= 0.2): 
        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        #mstats.register("std", np.std)
        #mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        self.pop, self.log = eaSimple(x, y, enhance_vector, self.pset, self.pop, self.toolbox, cxProb, mutProb, generation, 
                                            stats=mstats, halloffame=self.hof, verbose=False)
        self.normalizer = MinMaxScaler()
        self.normalizer.fit(self.transform_not_normalized(x))
        
    def transform_not_normalized(self, x):
        new_features = []
        for ind_num, ind in enumerate(self.hof):
            func = gp.compile(expr=ind, pset=self.pset)
            new_features.append([func(*record) for record in x])
        
        new_features = np.transpose(np.array(new_features))
        return new_features
        
    def transform(self, x):
        new_x = self.transform_not_normalized(x)
        return self.normalizer.transform(new_x)
        