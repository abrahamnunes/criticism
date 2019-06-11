# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
import pandas as pd
import scipy.stats as ss
import sklearn.metrics as skm
from criticism import metrics

# SIMULATORS
class NullClassifierSim(object):
    """ Simulates the performance of a null classifier """
    def __init__(self, alpha=0.5, n=100, metric=metrics.Kappa, rng=np.random.RandomState()):
        """
        Arguments:

            alpha: `float`. The proportion of positive classes in the dataset
            n: `int`. The number of observations in the dataset
            metric: `function`. A function that returns a classification performance metric given `alpha`, `beta`, `gamma` inputs
            rng: `numpy.random.RandomState`.
        """
        self.alpha = alpha
        self.n = n
        self.ncases = alpha*n
        self.ncontrols = (1-alpha)*n
        self.rng=rng
        self.sensitivity_prior = ss.beta(0.5*self.ncases, 0.5*self.ncontrols)
        self.specificity_prior = ss.beta(0.5*self.ncontrols, 0.5*self.ncases)
        self.metric = metric

    def __call__(self, nsamples=10000):
        """ Runs the simulation

        Arguments:

            nsamples: `int`. Number of samples to draw from the simulated distribution

        Returns:

            `ndarray(nsamples)`. Samples from the distribution on the classification statistic.
        """
        sensitivity = self.sensitivity_prior.rvs(size=nsamples, random_state=self.rng)
        specificity = self.specificity_prior.rvs(size=nsamples, random_state=self.rng)
        return self.metric(self.alpha, sensitivity, specificity)

class MultiNullClassifierSim(object):
    """ Simulates null classifier performance across several conditions with different sample sizes and case/control proportions """
    def __init__(self, alpha=[0.5], n=[100], metric=metrics.Kappa, rng=np.random.RandomState()):
        """
        Arguments:

            alpha: `list` or `ndarray(K)`. For each of the `K` conditions, the proportion of cases vs. controls
            n: `list` or `ndarray(K)`. For each of the `K` conditions, the number of samples total
            metric: `function`. The classification performance metric
            rng: `np.random.RandomState`
        """
        self.alpha = alpha
        self.n = n
        self.K = len(n)
        self.metric = metric
        self.rng = rng
        self._build_classifier_sim()

    def _build_classifier_sim(self):
        """ Constructs a list for iterating over the different conditions' simulators """
        self.sim = []
        for k in range(self.K):
            self.sim.append(NullClassifierSim(alpha = self.alpha[k], n=self.n[k], metric=self.metric, rng=self.rng))

    def __call__(self, nsamples=10000):
        """ Runs the simulation

        Arguments:

            nsamples: `int`. Number of samples to draw from the simulated distribution

        Returns:

            `ndarray((K, nsamples))`. Samples from the distribution on the classification statistic across conditions
        """
        out = np.stack([self.sim[i](nsamples) for i in range(self.K)])
        return out

class TestStatisticSim(object):
    """ Simulates the distribution of various test statistics for classification performance on cases with multiple conditions  """
    def __init__(self, alpha=[0.5], n=[100], metric=metrics.Kappa, tstat=np.mean, rng=np.random.RandomState()):
        """
        Arguments:

            alpha: `list` or `ndarray(K)`. For each of the `K` conditions, the proportion of cases vs. controls
            n: `list` or `ndarray(K)`. For each of the `K` conditions, the number of samples total
            metric: `function`. The classification performance metric
            tstat: `function` or `list` of functions. Test statistics to compute on simulated data
            rng: `np.random.RandomState`
        """
        self.multisim = MultiNullClassifierSim(alpha=alpha, n=n, metric=metric, rng=rng)
        self.tstat = tstat
        if type(tstat) == list:
            self.nstats = len(tstat)
        else:
            self.nstats = 1

    def __call__(self, nsamples=10000):
        sim_results = self.multisim(nsamples)
        out = np.empty((sim_results.shape[0], self.nstats))
        for j in range(self.nstats):
            out[:,j] = self.tstat[j](sim_results, axis=1)
        return out

class EmpiricalDistribution(object):
    """ Given a 1D dataset, constructs an empirical distribution """
    def __init__(self, data):
        """
        Arguments:

            data: `ndarray(nsamples)`. Vector of samples with which to construct the distribution

        """
        self.data = data
        self.nsamples = data.size
        self.mean_ = np.mean(data)
        self.median_ = np.median(data)
        self.std_ = np.std(data)
        self.stderr_ = self.std_/np.sqrt(self.nsamples)

    def confint(self, significance_threshold=0.05):
        """ Computes the `p*100` confidence interval on the data

        Arguments:

            significance_threshold: `0<float<1`. The statistical significance threshold

        Returns:

            `ndarray(2)`. Low and high values of the confidence interval

        """
        Zhalf = ss.norm(0, 1).ppf((1-significance_threshold)/2)
        dX = Zhalf*self.stderr_
        return np.array([self.mean_ - dX, self.mean_ + dX])

    def credint(self, p=0.95):
        """ Computes the `p*100` credible interval on the data

        Arguments:

            p: `0<float<1`. The range of the credible interval to compute

        Returns:

            `ndarray(2)`. Low and high values of the credible interval

        """
        dp = 100*(1-p)/2
        return np.percentile(self.data, [dp, 100-dp])

    def cdf(self, x):
        """ Value of the empirical cumulative distribution function at `x`

        Arguments:

            x: `float` or `list` or `ndarray(npoints)`. Points at which to evaluate the CDF

        Returns:

            `float` or `ndarray(npoints)`. Values of the CDF at the specified points
        """
        if type(x) == list or type(x) == np.ndarray:
            nlevels = len(x)
            out = np.stack([np.less_equal(self.data, x_i).sum()/self.nsamples for x_i in x])
        else:
            out = np.less_equal(self.data, x).sum()/self.nsamples
        return out

    def sf(self, x):
        """ Value of the empirical survival function at `x` (`sf=1-cdf`)

        Arguments:

            x: `float` or `list` or `ndarray(npoints)`. Points at which to evaluate the SF

        Returns:

            `float` or `ndarray(npoints)`. Values of the SF at the specified points
        """
        return 1 - self.cdf(x)

class SimulationScorer(object):
    """ Makes a classification performance scorer with 'significance' testing done by comparison against a null classifier
    """
    def __init__(self, metric='accuracy', nsamples=10000):
        """
        Arguments:

            metric: `{'accuracy', 'f1_score', 'cohen_kappa_score', 'matthews_corrcoef', 'sensitivity', 'specificity', 'ppv', 'npv'}`. Which metric to use
            nsamples: `int`. Number of samples to draw from the null distribution
        """
        self.simulation_metric, self.score_metric, self.stat_name = metrics.metricdict[metric]
        self.nsamples = nsamples

    def summary(self, round=3, credint_p=0.95):
        """ Creates a dataframe with the report summary

        Arguments:

            round: `int`. Number of decimal points to which values should be rounded
            credint_p: `0<float<1`. The percentage of the credible interval

        Returns:

            `pandas.DataFrame`

        """
        self.credint_ = self.probdist.credint(p=credint_p)
        self.results_ = pd.DataFrame({
            'Value': [self.score_],
            'P(t>Value)': [self.pvalue_],
            'Null Median': [self.probdist.median_],
            'Null CrI': ["(%s, %s)" %(np.round(self.credint_[0], round), np.round(self.credint_[1], round))],
            'N': [self.n_],
            'P(y=1)': [self.alpha_],
            'NSamples': [self.nsamples]}, index=None)
        self.results_.index = [self.stat_name]
        return self.results_

    def __call__(self, y, ypred, print_output=False):
        """ Runs the simulation-based scoring

        Arguments:

            y: `ndarray(n)`. The true classes
            y_pred: `ndarray(n)`. The predicted classes
            print_output: `bool`. Whether to print the results

        Returns:

            score_: `float`. The empirical score on the given classificaiton metric
            pvalue_: `float`. The number of samples from the null distribution that exceeded the measured statistic

        """
        self.n_ = y.size
        self.alpha_ = y.sum()/self.n_
        self.score_ = self.score_metric(y, ypred)
        self._null_simulator = NullClassifierSim(alpha=self.alpha_,
                                                 n=self.n_,
                                                 metric=self.simulation_metric)
        self.null_samples_ = self._null_simulator(nsamples=self.nsamples)
        self.probdist = EmpiricalDistribution(self.null_samples_)
        self.pvalue_ = self.probdist.sf(self.score_)
        return self.score_, self.pvalue_
