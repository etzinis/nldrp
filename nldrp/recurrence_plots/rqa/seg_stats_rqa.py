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
import nldrp.feature_extraction.stat_funcs as stats_funcs
import nldrp.feature_extraction.deltas as deltas

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

        self.phase_space_method = phase_space_method
        self.time_lag = time_lag
        self.embedding_dimension = embedding_dimension
        self.norm = norm
        self.thresh_method = thresh_method
        self.thresh = thresh
        self.l_min = l_min
        self.v_min = v_min
        self.w_min = w_min


    def frame_rqa_extraction(self,
                             frame):

        # configure the frame extractor as supposed to
        frame_rqa_extractor = rqa.RQA(
                phase_space_method=self.phase_space_method,
                time_lag=self.time_lag,
                embedding_dimension=self.embedding_dimension,
                norm=self.norm,
                thresh_method=self.thresh_method,
                thresh=self.thresh,
                l_min=self.l_min,
                v_min=self.v_min,
                w_min=self.w_min)

        # extract the rqa measures for this frame
        return frame_rqa_extractor.RQA_extraction(frame)


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

        frames_rqa = np.array(
            [self.frame_rqa_extraction(signal[st:st + frame_size])
             for st in frames_st_inds]
        )

        frames_rqa_d = deltas.compute(frames_rqa)

        print frames_rqa[5]
        print frames_rqa[4]

        rqa_stats = stats_funcs.compute(frames_rqa)
        rqa_d_stats = stats_funcs.compute(frames_rqa_d)

        whole_stats = np.concatenate([rqa_stats, rqa_d_stats], axis=0)

        return whole_stats


def test_performance(iterations=5):
    import time

    f0_list = np.random.uniform(low=40.0, high=700.0,
                                size=(iterations,))
    f0_list = np.sort(f0_list)
    fs_list = [8000, 16000, 44100]
    win_secs = [0.02, 0.04, 0.06]

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


def example_of_usege():
    frame_duration = 0.02
    frame_stride = 0.01
    fs = 44100
    s_len = int(0.3 * fs)
    signal = np.random.normal(0., 1., s_len)

    # signal = np.cos((2. * np.pi * 30 / fs) * np.arange(s_len))

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

    stats = seg_extr.extract(signal)
    print stats


if __name__ == '__main__':
    test_performance()

