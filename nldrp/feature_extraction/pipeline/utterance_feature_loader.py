"""!
\brief Utterance level feature loader pipeline for loading the
extracted features in appropriate format for utterance calssification
from scikit models and kfold--leave one out pipeline.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import numpy as np
import os
import sys
from sklearn.externals import joblib

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)
import nldrp.config


def get_args():
    """! Command line parser for Utterance level feature loader
    pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level feature loader pipeline' )
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


def load(config):
    exper_dat_name = ('{}-rqa-{}-tau-{}-{}-{}-{}-dur-{}-fs-{}'
                      '.dat'.format(
        config['dataset'],
        config['phase_space_method'],
        config['time_lag'],
        config['norm'],
        config['thresh_method'],
        config['thresh'],
        config['frame_duration'],
        config['fs']
    ))

    utterance_save_dir = os.path.join(config['save_dir'], 'utterance/')
    save_p = os.path.join(utterance_save_dir, exper_dat_name)

    return joblib.load(save_p)


def convert_2_numpy_per_utterance(dataset_dic):
    converted_dic = {}
    for spkr in dataset_dic:
        x_list = []
        y_list = []
        converted_dic[spkr] = {}
        for id, el_dic in dataset_dic[spkr].items():
            label = el_dic['y']
            feat_vec = el_dic['x']
            x_list.append(feat_vec)
            y_list.append(label)

        this_utt_array = np.array(x_list)
        converted_dic[spkr]['x']=this_utt_array
        converted_dic[spkr]['y']=y_list

    return converted_dic


def load_and_convert(config):

    loaded_dic = load(config)
    converted_dic = convert_2_numpy_per_utterance(loaded_dic)


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
    converted_feats_dic = load_and_convert(config)