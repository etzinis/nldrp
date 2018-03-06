"""!
\brief Utterance level classification schema by utilizing various
models for the configured experiment.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import numpy as np
import os
import sys

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, nldrp_dir)
import nldrp.feature_extraction.pipeline.utterance_feature_loader as \
    feature_loader
import nldrp.config


def generate_speaker_folds(features_dic):
    for te_speaker, te_data in features_dic.items():
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in features_dic.items():
            if tr_speaker == te_speaker:
                continue
            x_tr_list.append(tr_data['x'])
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        yield te_speaker, te_data['y'], X_tr, Y_tr


def loso(fusion_method, config):

    features_dic = feature_loader.load_and_convert(fusion_method, config)

    for X_te, Y_te, X_tr, Y_tr in generate_speaker_folds(features_dic):
        print X_te
        print len(Y_te)
        print len(Y_tr)


def get_args():
    """! Command line parser for Utterance level classification Leave
    one speaker out schema pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level classification Leave one '
                    'speaker out schema pipeline' )
    parser.add_argument("--dataset", type=str,
                        help="""The name of the dataset""",
                        required=True,
                        choices=['SAVEE'])
    parser.add_argument("-i", "--save_dir", type=str,
        help="""Where the corresponding binary file full of 
        data that will contain the dictionary for each speaker is 
        stored. 
        Another subdic for all the sentences with their ids  
        and a 1d numpy matrix for each one of them.
        """,
        default=nldrp.config.EXTRACTED_FEATURES_PATH )
    parser.add_argument("--features_fusion_method", type=str,
                        help="""Linear or RQA nonlinear features or 
                        their early 
                        fusion by concatenation""",
                        default='rqa',
                        choices=['rqa', 'linear','fusion'])
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
    parser.add_argument("-fs", type=float,
                        help="""Sampling frequency Hz""",
                        default=44100)
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
        'frame_stride':args.frame_duration / 2.0,
        'fs':args.fs
    }
    converted_feats_dic = loso(args.features_fusion_method,
                               config)
