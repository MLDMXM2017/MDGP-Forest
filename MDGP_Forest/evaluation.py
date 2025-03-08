from sklearn.metrics import accuracy_score,f1_score

def accuracy(y_true,y_pred):
    '''
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    float
        Call the evaluation function of sklearn to evaluate the accuracy of the prediction results.

    '''
    return accuracy_score(y_true,y_pred)

def f1_binary(y_true,y_pred):
    '''
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    float
        Call the evaluation function of sklearn to evaluate the f1_binary of the prediction results.

    '''
    f1=f1_score(y_true,y_pred,average="binary")
    return f1

def f1_micro(y_true,y_pred):
    '''
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    float
        Call the evaluation function of sklearn to evaluate the f1_micro of the prediction results.

    '''
    f1=f1_score(y_true,y_pred,average="micro")
    return f1

def f1_macro(y_true,y_pred):
    '''
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    float
        Call the evaluation function of sklearn to evaluate the f1_macro of the prediction results.

    '''
    f1=f1_score(y_true,y_pred,average="macro")
    return f1