import argparse
import os
import sys
from sklearn.externals import joblib
from progress.bar import ChargingBar
import time

nldrp_dir = os.path.join(
   os.path.dirname(os.path.realpath(__file__)),
   '../../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config
import nldrp.feature_extraction.pipeline.utterance_feat_extraction as\
    feat_extract

valid_thresh_methods = ["threshold",
                        "threshold_std",
                        "recurrence_rate"]

valid_norms = ["manhattan", "euclidean", "supremum"]

threshold_ratios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

frame_durations = [0.02, 0.03, 0.05]

taus = [1, 7]

for thres_method in valid_thresh_methods:
    for norm in valid_norms:
        for thres in threshold_ratios:
            for fd in frame_durations:
                configa={
                    'dataset':'SAVEE',
                    'cache_dir': '/tmp/',
                    'save_dir':nldrp.config.EXTRACTED_FEATURES_PATH,
                    'phase_space_method':'ad_hoc',
                    'time_lag':1,
                    'embedding_dimension':3,
                    'norm':norm,
                    'thresh_method':thres_method,
                    'thresh':thres,
                    'l_min':2,
                    'v_min':2,
                    'w_min':1,
                    'frame_duration':fd,
                    'frame_stride':fd / 2.0
                }
 
                try:
                    feat_extract.run(configa)
                except Exception as e:
                    print(e)
                    exit()
                    pprint.pprint(configa)
                    pass
