"""!
\brief Recurrence Quantification Analysis (RQA) feature extraction
for a given numpy array of a time series. Its a wrapper of pyunicorn
rqa corresponding module.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import os
import sys

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config as config

sys.path.insert(0, config.PYUNICORN_PATH)

import pyunicorn.timeseries.recurrence_plot as uni_rp


class RQA(object):
    """!
    \brief RQA analysis of a time series"""

    def __init__(self,
                 phase_space_method='ad-hoc',
                 time_lag=1,
                 embedding_dimension=3,
                 norm='euclidean',
                 recurrence_rate=None,
                 recurrence_thresh=None):

        if ((recurrence_rate is None and recurrence_thresh is None)
            or not (recurrence_rate is None
                    or recurrence_thresh is None)):
            raise ValueError(("Please specify only one of the "
                              "recurrence threshold xor recurrence "
                              "rate in order to construct the "
                              "recurrence plot"))
        elif recurrence_thresh is not None:
            if recurrence_thresh > 0.0 and recurrence_thresh < 1.0:
                self.r_thresh = recurrence_thresh
            else:
                raise ValueError('Recurrence Threshold <{}> not set '
                                 'into (0,1)'
                                 ''.format(recurrence_thresh))
        else:
            if recurrence_rate > 0.0 and recurrence_rate < 1.0:
                self.r_rate = recurrence_rate
            else:
                raise ValueError('Recurrence Rate <{}> not set '
                                 'into (0,1)'
                                 ''.format(recurrence_rate))

        valid_norms = ["manhattan", "euclidean", "supremum"]
        if norm not in valid_norms:
            raise ValueError(('Norm: {} is not valid. Pls try one of '
                              'the following: '.format(valid_norms)))
        else:
            self.norm = norm

        self.rps_method = phase_space_method
        self.tau = time_lag
        self.ed = embedding_dimension
