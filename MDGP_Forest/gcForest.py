import numpy as np
from layer import layer
from logger import get_logger
from k_fold_wrapper import KFoldWapper

from concurrent.futures import ProcessPoolExecutor

from GP.CategoryRelatedFC import CategoryRelatedFC

LOGGER = get_logger("gcForest")

def wrapper_trainAForest(args):
    return trainAForest(*args)

def trainAForest(layer_id, index, config, x_train, y_train, random_state):
    k_fold_est = KFoldWapper(layer_id, index, config, random_state=random_state)
    y_proba = k_fold_est.fit(x_train, y_train)
    return k_fold_est, y_proba

class gcForest(object):
    '''
    Implementation class of gcForest algorithm in the paper "Deep Forest".
    '''
    
    def __init__(self, config):
        '''
        This method will initialize according to the passed hyperparameter settings.
        
        Parameters
        ----------
        config : object
            This object contains the hyperparameter settings of the experiment.

        Returns
        -------
        None.

        '''
        self.random_state      = config["random_state"]
        self.max_layers        = config["max_layers"]
        self.early_stop_rounds = config["early_stop_rounds"]
        self.if_stacking       = config["if_stacking"]
        self.if_save_model     = config["if_save_model"]
        self.train_evaluation  = config["train_evaluation"]
        self.estimator_configs = config["estimator_configs"]
        self.layers = []
        self.crfc = None
  

    def fit(self, x_train, y_train, feature_num = 10, hardness_threshold = 0.95, pop_num=50, generation=20):
        '''
        Same as fit () of other classifiers, according to x_train and y_train training model.
        
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. 

        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        Returns
        -------
        None.

        '''
        
        x_train, n_feature, n_label = self.preprocess(x_train, y_train)
        
        estimator_num = len(self.estimator_configs)
        evaluate = self.train_evaluation
        best_layer_id = 0
        deepth = 0
        best_layer_evaluation = 0.0
        
        input_num = x_train.shape[1]
        feature_num = feature_num
        self.crfc = CategoryRelatedFC(input_num, categories_num=len(np.unique(y_train)))
        enhance_vector_len = 0
        y_probs = None
        
        while deepth < self.max_layers:
            y_train_probas = np.zeros((x_train.shape[0], n_label * len(self.estimator_configs)))
            
            current_layer = layer(deepth)
            LOGGER.info("-----------------------------------------layer-{}--------------------------------------------".format(current_layer.layer_id))
            LOGGER.info("The shape of x_train is {}".format(x_train.shape))
            
            if deepth != 0:
                self.crfc.fit(deepth, pop_num, feature_num, x_train, y_train, y_probs, enhance_vector_len, hardness_threshold, generation)
                new_x = self.crfc.transform(x_train, deepth, enhance_vector_len)
            else:
                new_x = x_train

            y_train_probas_avg = np.zeros((x_train.shape[0], n_label))
            
            params = [(current_layer.layer_id, i, self.estimator_configs[i].copy(), new_x, y_train, self.random_state) for i in range(estimator_num)]
            with ProcessPoolExecutor(max_workers=4) as pool:
                return_results = list(pool.map(wrapper_trainAForest, params))
            for i in range(estimator_num):
                k_fold_est = return_results[i][0]
                y_proba = return_results[i][1]
                current_layer.add_est(k_fold_est)
                y_train_probas[:, i * n_label:i * n_label + n_label] += y_proba
                y_train_probas_avg += y_proba
                
            y_train_probas_avg /= len(self.estimator_configs)
            label_tmp = self.category[np.argmax(y_train_probas_avg, axis=1)]
            current_evaluation = evaluate(y_train,label_tmp)

            if self.if_stacking:
                x_train = np.hstack((x_train, y_train_probas))
                enhance_vector_len += y_train_probas.shape[1]
            else:
                x_train = np.hstack((x_train[:, 0:n_feature], y_train_probas))
                enhance_vector_len = y_train_probas.shape[1]

            if current_evaluation > best_layer_evaluation:
                best_layer_id = current_layer.layer_id
                best_layer_evaluation = current_evaluation
            LOGGER.info("The evaluation[{}] of layer_{} is {:.4f}".format(evaluate.__name__, deepth, current_evaluation))

            y_probs = y_train_probas_avg
            self.layers.append(current_layer)

            if current_layer.layer_id - best_layer_id >= self.early_stop_rounds:
                self.layers = self.layers[0:best_layer_id + 1]
                LOGGER.info("training finish...")
                LOGGER.info("best_layer: {}, current_layer:{}, save layers: {}".format(best_layer_id, current_layer.layer_id, len(self.layers)))
                break

            deepth += 1

    def predict(self, x):
        '''
        Predict class for x.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        label : ndarray of shape (n_samples,)
            The predicted values.

        '''
        prob = self.predict_proba(x)
        label = self.category[np.argmax(prob,axis=1)]
        return label
    
    def predict_with_proba(self, x):
        '''
        Predict class for x.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        label : ndarray of shape (n_samples,)
            The predicted values.

        '''
        prob = self.predict_proba(x)
        label = self.category[np.argmax(prob,axis=1)]
        return label, prob

    def predict_proba(self, x):
        '''
        Predict class for x.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        x_test_proba : ndarray
            The prediction results are evaluated according to the evaluation function.

        '''
        x_test = x.copy()
        x_test = x_test.reshape((x.shape[0],-1))
        n_feature = x_test.shape[1]
        x_test_proba = None
        enhance_vector_len = 0

        for index in range(len(self.layers)):
            if index != 0:
                x_new = self.crfc.transform(x_test, index, enhance_vector_len)
            else:
                x_new = x_test 
            
            if index == len(self.layers) - 1:
                x_test_proba = self.layers[index]._predict_proba(x_new)
            else:
                x_test_proba = self.layers[index].predict_proba(x_new)
                if (not self.if_stacking):
                    x_test = x_test[:, 0:n_feature]
                    enhance_vector_len = x_test_proba.shape[1]
                else:
                    enhance_vector_len += x_test_proba.shape[1]
                x_test = np.hstack((x_test, x_test_proba))
        return x_test_proba


    def preprocess(self, x_train, y_train):
        '''
        Pre process the data and get some basic information of the data.
        
        Parameters
        ----------
        x_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            The raw input samples. 

        y_train : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        Returns
        -------
        x_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            The processe input samples.
            
        n_feature : int
            Number of features.
            
        n_label : int
            Number of labels.

        '''
        x_train = x_train.reshape((x_train.shape[0], -1))
        category = np.unique(y_train)
        self.category = category
        n_feature = x_train.shape[1]
        n_label = len(np.unique(y_train))
        LOGGER.info("Begin to train....")
        LOGGER.info("the shape of training samples: {}".format(x_train.shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking, self.if_save_model))
        return x_train, n_feature, n_label
    