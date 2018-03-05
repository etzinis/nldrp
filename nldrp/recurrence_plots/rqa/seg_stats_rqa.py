"""!
\brief Recurrence Quantification Analysis (RQA) feature extraction
for deltas and global statistics extraction for a segment. This
breaks to frames of the configured length (infered from duration in
seconds and the fs)

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import rqa


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
                 frame_ol_ratio=0.5,
                 fs=None):

        self.frame_duration=frame_duration
        self.frame_ol_ratio=frame_ol_ratio
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