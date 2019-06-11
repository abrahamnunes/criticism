import numpy as np
import sklearn.metrics as skm

# Empirical score
def TP(y, ypred):
    """ Computes the number of true positives in a classification run

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `int`. Number of true positives
    """
    return np.logical_and(np.equal(y, 1), np.equal(ypred, 1)).sum()

def TN(y, ypred):
    """ Computes the number of true negatives in a classification run

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `int`. Number of true negatives
    """
    return np.logical_and(np.equal(y, 0), np.equal(ypred, 0)).sum()

def FN(y, ypred):
    """ Computes the number of false negatives in a classification run

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `int`. Number of false negatives
    """
    return np.logical_and(np.equal(y, 1), np.equal(ypred, 0)).sum()

def FP(y, ypred):
    """ Computes the number of false positives in a classification run

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `int`. Number of false positives
    """
    return np.logical_and(np.equal(y, 0), np.equal(ypred, 1)).sum()

def sensitivity_score(y, ypred):
    """ Computes the empirical sensitivity score

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `float`. The classifier's sensitivity
    """
    tp = TP(y, ypred)
    fn = FN(y, ypred)
    return tp/(tp+fn)

def specificity_score(y, ypred):
    """ Computes the empirical specificity score

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `float`. The classifier's specificity
    """
    tn = TN(y, ypred)
    fp = FP(y, ypred)
    return tn/(tn+fp)

def ppv_score(y, ypred):
    """ Computes the empirical positive predictive value score

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `float`. The classifier's PPV
    """
    tp = TP(y, ypred)
    fp = FP(y, ypred)
    return tp/(tp+fp)

def npv_score(y, ypred):
    """ Computes the empirical negative predictive value score

    Arguments:

        y: `ndarray(nsamples)`. True classes
        ypred: `ndarray(nsamples)`. Predicted classes

    Returns:

        `float`. The classifier's NPV
    """
    tn = TN(y, ypred)
    fn = FN(y, ypred)
    return tn/(tn+fn)

# Simulation metrics
def Ptp(alpha, beta, gamma):
    """ The probability of true positives

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The probability of true positives
    """
    return alpha*beta

def Pfn(alpha, beta, gamma):
    """ The probability of false negatives

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The probability of false negatives
    """
    return alpha*(1-beta)

def Pfp(alpha, beta, gamma):
    """ The probability of false positives

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The probability of false positives
    """
    return (1-alpha)*(1-gamma)

def Ptn(alpha, beta, gamma):
    """ The probability of true negatives

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The probability of true negatives
    """
    return (1-alpha)*gamma

# Composite classification metrics
def Sensitivity(alpha, beta, gamma):
    """ Computes the expected sensitivity. This is a dummy wrapper that just returns the value of `beta`. Exists only for compatability

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's sensitivity

    """
    return beta

def Specificity(alpha, beta, gamma):
    """ Computes the expected specificity. This is a dummy wrapper that just returns the value of `gamma`. Exists only for compatability

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's specificity

    """
    return gamma

def PPV(alpha, beta, gamma):
    """ Computes the expected positive predictive value.

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's positive predictive value

    """
    tp = Ptp(alpha, beta, gamma)
    fp = Pfp(alpha, beta, gamma)
    return tp/(tp+fp)

def NPV(alpha, beta, gamma):
    """ Computes the expected negative predictive value.

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's negative predictive value

    """
    tn = Ptn(alpha, beta, gamma)
    fn = Pfn(alpha, beta, gamma)
    return tn/(tn+fn)

def Accuracy(alpha, beta, gamma):
    """ Computes the expected accuracy.

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's accuracy

    """
    return Ptp(alpha, beta, gamma) + Ptn(alpha, beta, gamma)

def PChance(alpha, beta, gamma):
    """ Computes the probability of chance agreement

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's probability of chance agreement

    """
    tp = Ptp(alpha, beta, gamma)
    fp = Pfp(alpha, beta, gamma)
    tn = Ptn(alpha, beta, gamma)
    fn = Pfn(alpha, beta, gamma)
    return (tp + fn)*(tp + fp) + (tn + fn)*(tn + fp)

def Kappa(alpha, beta, gamma):
    """ Computes the Cohen's Kappa metric

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's kappa score

    """
    p_obs = Accuracy(alpha, beta, gamma)
    p_chance = PChance(alpha, beta, gamma)
    return (p_obs-p_chance)/(1-p_chance)

def F1Score(alpha, beta, gamma):
    """ Computes the expected F1Score.

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's F1Score

    """
    precision = PPV(alpha, beta, gamma)
    recall = beta
    return 2*((precision*recall)/(precision + recall))

def MCC(alpha, beta, gamma):
    """ Computes the expected Matthews Correlation Coefficient.

    Arguments:

        alpha: `float`. The rate of positive class in the sample
        beta: `float`. The classifier's sensitivity
        gamma: `float`. The classifier's specificity

    Returns:

        `float`. The classifier's MCC

    """
    tp = Ptp(alpha, beta, gamma)
    fp = Pfp(alpha, beta, gamma)
    tn = Ptn(alpha, beta, gamma)
    fn = Pfn(alpha, beta, gamma)
    a = tp*tn - fp*fn
    b = np.sqrt((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp))
    return a/b



# Metrics dictionary
metricdict = {
    'accuracy': [Accuracy, skm.accuracy_score, 'Accuracy'],
    'f1_score': [F1Score, skm.f1_score, 'F1'],
    'cohen_kappa_score': [Kappa, skm.cohen_kappa_score, 'Kappa'],
    'matthews_corrcoef': [MCC, skm.matthews_corrcoef, 'MCC'],
    'sensitivity': [Sensitivity, sensitivity_score, 'Sensitivity'],
    'specificity': [Specificity, specificity_score, 'Specificity'],
    'ppv': [PPV, ppv_score, 'PPV'],
    'npv': [NPV, npv_score, 'NPV']}
