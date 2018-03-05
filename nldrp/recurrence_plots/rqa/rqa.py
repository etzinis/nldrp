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
                 phase_space_method='ad_hoc',
                 time_lag=1,
                 embedding_dimension=3,
                 norm='euclidean',
                 thresh_method='recurrence_thresh',
                 thresh=0.1):

        valid_thresh_methods = ["recurrence_thresh",
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


if __name__ == "__main__":
    rqa_obj = RQA(phase_space_method='ad_hoc',
                  time_lag=1,
                  embedding_dimension=3,
                  norm='euclidean',
                  thresh_method='recurrence_thresh',
                  thresh=0.1)