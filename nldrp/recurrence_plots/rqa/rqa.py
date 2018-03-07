"""!
\brief Recurrence Quantification Analysis (RQA) feature extraction
for a given numpy array of a time series. Its a wrapper of pyunicorn
rqa corresponding module.

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

#sys.path.insert(0, config.PYUNICORN_PATH)

import pyunicorn.timeseries.recurrence_plot as uni_rp
import nldrp.phase_space.reconstruct.rps as rps


class RQA(object):
    """!
    \brief RQA analysis of a time series"""

    def __init__(self,
                 phase_space_method='ad_hoc',
                 time_lag=1,
                 embedding_dimension=3,
                 norm='euclidean',
                 thresh_method='threshold',
                 thresh=0.1,
                 l_min=2,
                 v_min=2,
                 w_min=1):

        valid_thresh_methods = ["threshold",
                                "threshold_std",
                                "recurrence_rate"]

        if thresh_method not in valid_thresh_methods:
            raise ValueError(("Please specify only one of the "
                              "recurrence threshold, recurrence "
                              "rate or threshold_std in order to "
                              "construct the recurrence plot"))
        else:
            if thresh > 0.0 and thresh < 1.0:
                self.r_config = {thresh_method:thresh}
            else:
                raise ValueError('Recurrence Rate <{}> not set '
                                 'into (0,1)'
                                 ''.format(thresh))

        valid_norms = ["manhattan", "euclidean", "supremum"]
        if norm not in valid_norms:
            raise ValueError(('Norm: {} is not valid. Pls try one of '
                              'the following: {}'.format(
                              norm, valid_norms)))
        else:
            self.norm = norm

        valid_phase_space_methods = ['ad_hoc', 'ami_ffn']
        if phase_space_method not in valid_phase_space_methods:
            raise ValueError(('Phase Space Method: {} is not valid. '
                              'Pls try one of '
                              'the following: {}'.format(
                              phase_space_method,
                              valid_phase_space_methods)))
        else:
            self.rps_method = phase_space_method
            self.tau = time_lag
            self.ed = embedding_dimension

        self.l_min = l_min
        self.v_min = v_min
        self.w_min = w_min


    def reconstruct_phase_space(self, x):
        """!
        \brief Phase space reconstruction wrapper for enabling
        various methods of Phase space reconstruction for 1d time
        series given (x)"""

        if self.rps_method == 'ad_hoc':
            return rps.cython_RPS(x, self.tau, self.ed)
        else:
            return rps.ami_RPS(x, ed=3)


    def config_binary_RP(self, x):
        """!
        \brief Wrapper for unicorn binary recurrence plot computation
        by using the configured variables and returning them.

        \warning It is assumed that the given 1d time series x is
        already normalized and will be embedded in its phase space as
        configured inside the class constructor."""

        embedded_x = self.reconstruct_phase_space(x)

        rp_obj = uni_rp.RecurrencePlot(embedded_x,
                                       silence_level=10,
                                       metric=self.norm,
                                       **self.r_config)

        return rp_obj


    def RQA_extraction(self, x, n_features=12):
        """!
        \brief Wrapping up various handcrafted features from a binary
        RP corresponding to the given time series and more
        specifically the embedded time series as configured in the
        constructor of the class."""

        rp_obj = self.config_binary_RP(x)

        res = np.empty(n_features)

        res[0] = rp_obj.recurrence_rate()
        # print 'RR    : {}'.format(t)
        res[1] = rp_obj.max_diaglength()
        # print 'L_max : {}'.format(t)
        res[2] = rp_obj.determinism(l_min=self.l_min)
        # print 'DET   : {}'.format(t)
        res[3] = rp_obj.average_diaglength(l_min=self.l_min)
        # print 'L     : {}'.format(t)
        res[4] = rp_obj.diag_entropy(l_min=self.l_min)
        # print 'ENTR  : {}'.format(t)
        res[5] = rp_obj.max_vertlength()
        # print 'V_max : {}'.format(t)
        res[6] = rp_obj.laminarity(v_min=self.v_min)
        # print 'LAM   : {}'.format(t)
        res[7] = rp_obj.average_vertlength(v_min=self.v_min)
        # print 'TT    : {}'.format(t)
        res[8] = rp_obj.vert_entropy(v_min=self.v_min)
        # print 'VLiENT: {}'.format(t)
        res[9] = rp_obj.max_white_vertlength()
        # print 'W_maxV: {}'.format(t)
        res[10] = rp_obj.average_white_vertlength(w_min=self.w_min)
        # print 'W_avgV: {}'.format(t)
        res[11] = rp_obj.white_vert_entropy(w_min=self.w_min)
        # print 'W_entV: {}'.format(t)

        return res


def test_performance(iterations=1000):
    import numpy as np
    import time

    f0_list = np.random.uniform(low=40.0, high=700.0,
                                size=(iterations,))
    f0_list = np.sort(f0_list)
    fs_list = [8000, 16000, 44100]
    win_secs = 0.02

    tau = 1
    ed = 3

    valid_thresh_methods = ["threshold",
                            "threshold_std",
                            "recurrence_rate"]

    print '=' * 5 + ' RQA Performance Testing ' + '=' * 5

    for fs in fs_list:
        total_time = dict([(v,0.0) for v in valid_thresh_methods])

        win_samples = int(win_secs * fs)
        print '\n\n' + '~' * 5 + ' Testing for Fs={} Samples={}  ' \
                                 ''.format(
            fs, win_samples) + '~' * 5

        for f0 in f0_list:
            x = np.cos(
                (2. * np.pi * f0 / fs) * np.arange(win_samples))

            for v in total_time:

                before = time.time()
                rqa_obj = RQA(phase_space_method='ad_hoc',
                              time_lag=1,
                              embedding_dimension=3,
                              norm='euclidean',
                              thresh_method=v,
                              thresh=0.1)
                bin_rp_obj = rqa_obj.RQA_extraction(x)
                now = time.time()
                total_time[v] += now - before

                # import matplotlib.pyplot as plt
                # # plt.imshow(bin_rp)
                # plt.imshow(bin_rp)

        for k, v in total_time.items():
            print (">Total Time: {} for {} frames, RP Class: "
                   "{}".format(v, iterations, k))


def example_of_usage(x):
    rqa_obj = RQA(phase_space_method='ad_hoc',
                  time_lag=1,
                  embedding_dimension=3,
                  norm='euclidean',
                  thresh_method='threshold',
                  thresh=0.1)

    return rqa_obj.binary_recurrence_plot(x)

if __name__ == "__main__":
    test_performance(iterations=100)
