"""!
\brief Recurrence Quantification Analysis (RQA) feature extraction
for deltas and global statistics extraction for a segment. This
breaks to frames of the configured length (infered from duration in
seconds and the fs)

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import rqa
import numpy as np
import os
import sys

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.feature_extraction.frame_breaker as frame_breaker


class SegmentRQAStatistics(object):

    def __init__(self,
                 phase_space_method='ad_hoc',
                 time_lag=1,
                 embedding_dimension=3,
                 norm='euclidean',
                 thresh_method='threshold',
                 thresh=0.1,
                 l_min=2,
                 v_min=2,
                 w_min=1,
                 frame_duration=0.02,
                 frame_stride=0.01,
                 fs=None):

        self.frame_duration=frame_duration
        self.frame_stride=frame_stride
        self.fs=fs

        # configure the frame extractor as supposed to
        self.rqa_extractor = rqa.RQA(
                phase_space_method=phase_space_method,
                time_lag=time_lag,
                embedding_dimension=embedding_dimension,
                norm=norm,
                thresh_method=thresh_method,
                thresh=thresh,
                l_min=l_min,
                v_min=v_min,
                w_min=w_min)

    def extract(self,
                signal):
        """!
        \brief Extract the Statistics from RQA frame measures for the
        given signal which corresponds to an audio segment"""

        frame_info = frame_breaker.get_frames_start_indices(
                                   signal,
                                   self.fs,
                                   frame_duration=self.frame_duration,
                                   frame_stride=self.frame_stride)
        frames_st_inds, frame_size, frame_step = frame_info

        frame_rqa = np.array(
            [self.rqa_extractor.RQA_extraction(
             signal[x:x + frame_size])
             for x in frames_st_inds]
        )

        print frame_rqa.shape


if __name__ == '__main__':

    frame_duration = 0.02
    frame_stride = 0.01
    fs=44100
    s_len=int(3.0*fs)
    signal = np.random.normal(0., 1., s_len)

    seg_extr = SegmentRQAStatistics(
                 phase_space_method='ad_hoc',
                 time_lag=1,
                 embedding_dimension=3,
                 norm='euclidean',
                 thresh_method='threshold',
                 thresh=0.1,
                 l_min=2,
                 v_min=2,
                 w_min=1,
                 frame_duration=frame_duration,
                 frame_stride=frame_stride,
                 fs=fs)

    seg_extr.extract(signal)
