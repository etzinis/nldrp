"""! 
\brief Reconstruct the Phase Space of a Signal
\details RP extractors in Factory class format for better abstraction

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import numpy as np
import os
import sys

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config as config

sys.path.insert(0, config.PYUNICORN_PATH)

import pyunicorn.timeseries.recurrence_plot as unicorn_rp
import nldrp.recurrence_plots.utils.distance_matrix as dutil


class RP(object):
    """Computation of various RPs"""

    @staticmethod
    def factory(RP_name,
                maximum_distance=None,
                thresh=0.1,
                norm='euclidean'):
        """!
        \brief This constructor checks whether all the parameters 
        have been set appropriatelly and finally set them by also 
        returning the appropriate implementation class. 
        Essentially we get an abstraction for the RP extraction 
        as the extractor is set internally."""

        if RP_name == 'continuous':
            return ContinuousRP(maximum_distance)
        elif RP_name == 'binary':
            return BinaryRP(thresh, maximum_distance)
        elif RP_name == 'unicorn_binary':
            return UnicornBinaryPlot(thresh,
                                     norm=norm)
        else:
            valid_RP_names = ['binary', 'continuous', 'unicorn_binary']
            raise NotImplementedError(('Recurrence Plot Name: <{}> '
                                       'is not a valid RP class name. Please use one of the '
                                       'following: {}'.format(RP_name,
                                                              valid_RP_names)))


class ContinuousRP(RP):
    """docstring for ContinuousRP"""

    def __init__(self, maximum):

        self.maximum = maximum

    def extract(self,
                X,
                norm='euclidean',
                n_jobs=1):
        """!
        \brief compute the normalized RP for the X representation

        \param X (\a numpy matrix) with size (n_samples, n_features)

        \returns rp (\a numpy matrix) representing the RP of the 
        representation in continuous form with size:
        (n_samples, n_samples)"""

        d = dutil.compute_distance_matrix(X,
                                          norm=norm,
                                          n_jobs=n_jobs
                                          )

        if self.maximum is None:
            this_maxima = np.amax(d)
        else:
            this_maxima = self.maximum

        cont = np.divide(d, this_maxima)

        return cont


class BinaryRP(object):
    """docstring for BinaryRP"""

    def __init__(self, thresh, maximum):
        if thresh > 0.0 and thresh < 1.0:
            self.thresh = thresh
        else:
            raise ValueError('Threshold <{}> not set into (0,1)'
                             ''.format(thresh))
        self.maximum = maximum

    def extract(self,
                X,
                norm='euclidean',
                n_jobs=1):
        """!
        \brief compute the normalized RP for the X representation

        \param X (\a numpy matrix) with size (n_samples, n_features)

        \returns rp (\a numpy matrix) representing the RP of the 
        representation in continuous form with size:
        (n_samples, n_samples)"""

        d = dutil.compute_distance_matrix(X,
                                          norm=norm,
                                          n_jobs=n_jobs
                                          )

        if self.maximum is None:
            this_maxima = np.amax(d)
        else:
            this_maxima = self.maximum

        cont = np.divide(d, this_maxima)

        return cont < self.thresh


class UnicornBinaryPlot(object):
    """!
    \brief Binary Recurrence Plot Wrapper Class for the efficient
    computation of RP by using pyunicorn"""

    def __init__(self,
                 threshold,
                 norm='euclidean',
                 **rp_kargs):
        if threshold > 0.0 and threshold < 1.0:
            self.threshold = threshold
        else:
            raise ValueError('Threshold <{}> not set into (0,1)'
                             ''.format(threshold))

        valid_norms = ["manhattan", "euclidean", "supremum"]
        if norm not in valid_norms:
            raise ValueError(('Norm: {} is not valid. Pls try one of '
                              'the following: '.format(valid_norms)))
        else:
            self.norm = norm

        # self.rp_kwargs = **{'thresh'}

    def extractRP(self, X):

        rp_obj = unicorn_rp.RecurrencePlot(X,
                                           silence_level=10,
                                           metric=self.norm,
                                           threshold=0.1)

        return rp_obj.recurrence_matrix()


def test_performance(iterations=1000):
    import nldrp.phase_space.reconstruct.rps as rps
    import time

    f0_list = np.random.uniform(low=40.0, high=700.0,
                                size=(iterations,))
    f0_list = np.sort(f0_list)
    fs_list = [8000, 16000, 44100]
    win_secs = 0.02

    tau = 1
    ed = 3

    cont_constr = RP.factory('continuous')
    bin_constr = RP.factory('binary')
    uni_bin_constr = RP.factory('unicorn_binary')

    print '=' * 5 + ' Recur. Plots Performance Testing ' + '=' * 5

    for fs in fs_list:
        total_time = {'Continuous RP': 0.0,
                      'Binary RP': 0.0,
                      'Unicorn Binary RP': 0.0}
        win_samples = int(win_secs * fs)
        print '\n\n' + '~' * 5 + ' Testing for Fs={} Samples={} '.format(
            fs, win_samples) + '~' * 5

        for f0 in f0_list:
            x = np.cos((2. * np.pi * f0 / fs) * np.arange(win_samples))

            phase_s = rps.rps(x, tau, ed)

            before = time.time()
            cont_r_plot = cont_constr.extract(phase_s,
                                              norm='euclidean',
                                              n_jobs=1)
            now = time.time()
            total_time['Continuous RP'] += now - before

            before = time.time()
            bin_r_plot = bin_constr.extract(phase_s,
                                            norm='euclidean',
                                            n_jobs=1)
            now = time.time()
            total_time['Binary RP'] += now - before

            before = time.time()
            uni_bin_r_plot = uni_bin_constr.extractRP(phase_s)
            now = time.time()
            total_time['Unicorn Binary RP'] += now - before

            # import matplotlib.pyplot as plt
            # # plt.imshow(cont_r_plot)
            # plt.imshow(bin_r_plot)
            # plt.show()
            # plt.imshow(uni_bin_r_plot)
            # plt.show()

        for k, v in total_time.items():
            print (">Total Time: {} for {} frames, RP Class: "
                   "{}".format(v, iterations, k))


if __name__ == "__main__":
    test_performance(iterations=1000)
