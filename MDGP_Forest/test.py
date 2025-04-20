# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:22:15 2023

@author: 12207
"""

from multiprocessing import freeze_support  

from gcForest import gcForest

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score

from evaluation import accuracy,f1_binary,f1_macro,f1_micro

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

def get_config():
    config = {}
    config["random_state"]=0
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_macro   ##accuracy,f1_binary,f1_macro,f1_micro
    config["estimator_configs"]=[]
    for i in range(2):
        config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"max_depth":None, "n_jobs": -1})
    for i in range(2):
        config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"max_depth":None, "n_jobs": -1})
    return config

if __name__ == '__main__':
    freeze_support()
    
    x, y = load_iris(return_X_y=True)
    scaler = MinMaxScaler()  
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    config = get_config()
    gc = gcForest(config)
    feature_num = 10
    hardness_threshold = 0.05
    pop_num = 50 
    generation = 20
    cxProb = 0.5
    mutProb = 0.2
    gc.fit(x_train, y_train, feature_num, hardness_threshold, pop_num, generation, cxProb, mutProb)
    
    y_pred, y_prob = gc.predict_with_proba(x_test)
    f1_ma = f1_score(y_test, y_pred, average='macro')
    g_mean_ma = geometric_mean_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"F1 Score (Macro Average): {f1_ma:.4f}")
    print(f"Geometric Mean Score (Macro Average): {g_mean_ma:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")