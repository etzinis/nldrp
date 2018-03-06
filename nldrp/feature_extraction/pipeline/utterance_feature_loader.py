"""!
\brief Utterance level feature loader pipeline for loading the
extracted features in appropriate format for utterance calssification
from scikit models and kfold--leave one out pipeline.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import os
import sys
from sklearn.externals import joblib

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)


def get_args():
    """! Command line parser for Utterance level feature loader
    pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level feature laoder pipeline' )
    parser.add_argument("--cache_dir", type=str,
        help="""Directory which would be available to store some 
        binary files for quicker load of dataset""",
        default='/tmp/')
    parser.add_argument("--dataset", type=str,
                        help="""The name of the dataset""",
                        required=True,
                        choices=['SAVEE'])
    parser.add_argument("-o", "--save_dir", type=str,
        help="""Where to store the corresponding binary file full of 
        data that will contain the dictionary for each speaker. 
        Another subdic for all the sentences with their ids  
        and a 1d numpy matrix for each one of them.
        """,
        default=nldrp.config.EXTRACTED_FEATURES_PATH )
    parser.add_argument("-tau", type=int,
                        help="""Time Delay Ad-hoc""",
                        default=1)
    parser.add_argument("--tau_est_method", type=str,
                        help="""How to estimate Time Delay (Using 
                        an adhoc value as set or estimate AMI per 
                        frame?)""",
                        default='ad_hoc',
                        choices=['ad_hoc', 'ami'])
    parser.add_argument("-norm", type=str,
                        help="""Norm for computing in RPs""",
                        default='euclidean',
                        choices=["manhattan", "euclidean", "supremum"])
    parser.add_argument("--thresh_method", type=str,
                        help="""How to threshold RPs""",
                        default='threshold',
                        choices=["threshold",
                                "threshold_std",
                                "recurrence_rate"])
    parser.add_argument("-thresh", type=float,
                        help="""Value of threshold in (0,1)""",
                        default=0.1)
    parser.add_argument("--frame_duration", type=float,
                        help="""Frame duration in seconds""",
                        default=0.02)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    config = {
        'dataset':args.dataset,
        'cache_dir':args.cache_dir,
        'save_dir':args.save_dir,
        'phase_space_method':args.tau_est_method,
        'time_lag':args.tau,
        'embedding_dimension':3,
        'norm':args.norm,
        'thresh_method':args.thresh_method,
        'thresh':args.thresh,
        'l_min':2,
        'v_min':2,
        'w_min':1,
        'frame_duration':args.frame_duration,
        'frame_stride':args.frame_duration / 2.0
    }
    run(config)