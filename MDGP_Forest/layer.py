import numpy as np 

class layer(object):
    '''
    The class that records the classifiers and information of each level of gcForest cascade.
    '''
    
    def __init__(self,layer_id):
        '''
        This method will instantiate a Layer class.
        
        Parameters
        ----------
        layer_id : int
            Serial number of the current cascade level.

        Returns
        -------
        None.

        '''
        self.layer_id   = layer_id
        self.estimators = []
    
    def add_est(self, estimator):
        '''
        Add a new classifier to the current layer.

        Parameters
        ----------
        estimator : object
            The classifier object to be added.

        Returns
        -------
        None.

        '''
        if estimator != None:
            self.estimators.append(estimator)

    def predict_proba(self,x):
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
        proba : ndarray
            The prediction results are evaluated according to the evaluation function.

        '''
        proba = None
        for est in self.estimators:
            proba = est.predict_proba(x) if proba is None else np.hstack((proba,est.predict_proba(x)))
        return proba
    
    def _predict_proba(self, x_test):
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
            proba = est.predict_proba(x_test) if proba is None else proba + est.predict_proba(x_test)
        proba /= len(self.estimators)
        return proba