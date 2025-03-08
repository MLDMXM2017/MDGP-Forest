from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from logger import get_logger
'''
此处导入新的包
'''

LOGGER_2 = get_logger("KFoldWrapper")

class KFoldWapper(object):
    '''
    Use KFold packaging classifier for training and verification.
    '''
    
    def __init__(self, layer_id, index, config, random_state):
        '''
        

        Parameters
        ----------
        layer_id : int
            Serial number of the current cascade level.
            
        index : int
            Serial number of the classifier in the cascade.
            
        config : object
            This object contains the hyperparameter settings of the experiment.
            
        random_state : int, RandomState instance or None, default=None
            Controls the random seed given to each Tree estimator at each
            boosting iteration.
            In addition, it controls the random permutation of the features at
            each split (see Notes for more details).
            It also controls the random spliting of the training data to obtain a
            validation set if `n_iter_no_change` is not None.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        None.

        '''
        self.config = config
        self.name = "layer_{}, estimstor_{}, {}".format(layer_id, index, self.config["type"])
        if random_state is not None:
            self.random_state = (random_state + hash(self.name)) % 1000000007
        else:
            self.random_state = None
        self.n_fold = self.config["n_fold"]
        self.estimators = [None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class = globals()[self.config["type"]]
        self.config.pop("type")
    
    def _init_estimator(self):
        '''
        Initialize a classifier in config.

        Returns
        -------
        object
            Classname of classifier.

        '''
        estimator_args = self.config
        est_args = estimator_args.copy()
        return self.estimator_class(**est_args)
    
    def fit(self, x, y):
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
        y_probas : : ndarray
            The prediction results are evaluated according to the evaluation function.

        '''
        skf = StratifiedKFold(n_splits = self.n_fold, shuffle=True, random_state=self.random_state)
        cv = [(t,v) for (t,v) in skf.split(x,y)]
        
        n_label = len(np.unique(y))
        y_probas = np.zeros((x.shape[0], n_label))

        for k in range(self.n_fold):
            est = self._init_estimator()
            train_id, val_id = cv[k]
            est.fit(x[train_id], y[train_id])
            y_proba = est.predict_proba(x[val_id])
            y_pred = est.predict(x[val_id])
            LOGGER_2.info("{}, n_fold_{},Accuracy={:.4f}, f1_score={:.4f}".format(self.name, k, accuracy_score(y[val_id], y_pred), f1_score(y[val_id], y_pred, average="macro")))
            y_probas[val_id] += y_proba
            self.estimators[k] = est
        LOGGER_2.info("{}, {},Accuracy={:.4f}, f1_score={:.4f}".format(self.name, "wrapper", accuracy_score(y, np.argmax(y_probas, axis=1)), f1_score(y, np.argmax(y_probas, axis=1), average="macro")))
        LOGGER_2.info("----------")
        return y_probas

    def predict_proba(self, x_test):
        '''
        Predict class for x_test.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        proba : ndarray
            The prediction results are evaluated according to the evaluation function.

        '''
        proba = None
        for est in self.estimators:
            if proba is None:
                proba = est.predict_proba(x_test)
            else:
                proba += est.predict_proba(x_test)
        proba /= self.n_fold
        return proba