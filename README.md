# MDGP-Forest
This repository contains the ​​Python implementation​​ of ​**​MDGP-Forest​**​ from the paper:

​​"MDGP-Forest: A Novel Deep Forest for Multi-class Imbalanced Learning Based on Multi-class Disassembly and Feature Construction Enhanced by Genetic Programming"​

# Requirements
The Python environment and versions of dependency packages used in this implementation are as follows:
```
Python==3.7.16
scikit-learn==1.0.2
imbalanced-learn==0.10.1
deap==1.4.1
numpy==1.21.6
```

# Usage
We provide a demo for MDGP-Forest in the test.py file, using the Iris dataset loaded via sklearn.load_iris(). If you wish to use your own dataset, you need to preprocess it into the same format as the return value of load_iris().
```
python test.py
```

# Hyperparameter Settings
The table below shows the ​​mapping between variables in test.py and the hyperparameters of MDGP-Forest​​. You can modify these variables to configure the hyperparameters. The settings in test.py represent the ​​default hyperparameter values​​ for MDGP-Forest.
| Variables | Hyperparameters | Values |
| --- | --- | --- |
| config | Deep Forest Configuration | Refer to test.py |
| hardness_threshold | Hardness Threshold | 0.05 |
| pop_num | GP Population Size | 50 | 
| generation | GP Maximum Generations | 20 |
| feature_num | Number of Features Constructed per Population | 10 |
| cxProb | GP Crossover Probability | 0.5 |
| mutProb | GP Mutation Probability | 0.2 |

# Cite This Repository
If you use this code in your research, please cite both the ​​original paper​​ and this ​​code repository​​:
Paper Citation (BibTeX):
```

```
Code Repository
```
@software{MDGPForest_Code,
  author = {Zhikai Lin},
  title = {MDGP-Forest},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/MLDMXM2017/MDGP-Forest.git}}
}
```
