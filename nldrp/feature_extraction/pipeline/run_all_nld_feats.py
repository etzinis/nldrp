import argparse
import os
import sys
from sklearn.externals import joblib
from progress.bar import ChargingBar
import time

nldrp_dir = os.path.join(
   os.path.dirname(os.path.realpath(__file__)),
   '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config
import nldrp.feature_extraction.pipeline.utterance_feat_extraction as\
    feat_extract

valid_thresh_methods = ["threshold",
                        "threshold_std",
                        "recurrence_rate"]

valid_norms = ["manhattan", "euclidean"]

threshold_ratios = [0.1, 0.15, 0.2]

frame_durations = [0.02, 0.03]

taus = [7]

search_space = (len(valid_thresh_methods) *
                len(threshold_ratios) *
                len(frame_durations) *
                len(valid_norms))

def main(args):
    cnt = 1
    for thres_method in valid_thresh_methods:
        for norm in valid_norms:
            for thres in threshold_ratios:
                for fd in frame_durations:
                    print "Extracting {}/{}...".format(cnt, search_space)
                    cnt += 1
                    configa={
                        'dataset':args.dataset,
                        'cache_dir': '/tmp/',
                        'save_dir':'/home/thymios/Desktop/Research'
                                   '/all_'+args.dataset+'_features/rqa/',
                        'phase_space_method':'ad_hoc',
                        'time_lag':7,
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


def get_args():
    """! Command line parser for Utterance level feature extraction
    pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level feature extraction pipeline' )
    parser.add_argument("--cache_dir", type=str,
        help="""Directory which would be available to store some 
        binary files for quicker load of dataset""",
        default='/tmp/')
    parser.add_argument("--dataset", type=str,
                        help="""The name of the dataset""",
                        required=True,
                        choices=['SAVEE',
                                 'IEMOCAP',
                                 'BERLIN'])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    main(args)
