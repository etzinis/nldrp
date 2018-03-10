"""!
\brief Utterance Level Model Optimizer

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
from sklearn.externals import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def compute_metrics(Y_predicted, Y_true):
    uw_f1 = f1_score(Y_predicted, Y_true, average='macro')
    w_acc = accuracy_score(Y_predicted, Y_true)
    cmat = confusion_matrix(Y_true, Y_predicted)
    with np.errstate(divide='ignore'):
        uw_acc = (cmat.diagonal() / (1.0 * cmat.sum(axis=1) + 1e-6
                                     )).mean()
        if np.isnan(uw_acc):
            uw_acc = 0.

    metrics_l = [('uw_f1', uw_f1),
                 ('uw_acc', uw_acc),
                 ('w_acc', w_acc)]

    metric_dic = dict(metrics_l)
    return metric_dic


def loso_with_best_models(features_dic):
    pass
    print yolo


def fuse_all_configurations(list_of_paths):

    try:
        feat_p = list_of_paths.pop(0)
        final_data_dic = joblib.load(feat_p)
    except Exception as e:
        print "At least one file path is required"
        raise e

    while list_of_paths:
        feat_p = list_of_paths.pop(0)
        temp_dic = joblib.load(feat_p)
        try:
            for spkr in temp_dic:
                for id, el_dic in temp_dic[spkr].items():
                    assert el_dic['y'] == final_data_dic[spkr][id]['y']
                    prev_vec = final_data_dic[spkr][id]['x']
                    this_vec = el_dic['x']
                    new_vec = np.concatenate([prev_vec, this_vec],
                                             axis=0)
                    final_data_dic[spkr][id]['x'] = new_vec
        except Exception as e:
            print "Failed to update the Fused dictionary"
            raise e

    fused_converted_dic = convert_2_numpy_per_utterance(final_data_dic)
    return evaluate_loso(fused_converted_dic)


def get_args():
    """! Command line parser for Utterance level classification Leave
    one speaker out schema pipeline -- Find Best Models"""
    parser = argparse.ArgumentParser(
        description='Utterance level classification Leave one '
                    'speaker out schema pipeline -- Find Best Models' )
    parser.add_argument('-i', '--input_features_paths', nargs='+',
                        help='File paths of the features you want to '
                             'concatenate and the classify')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    loso_with_best_models(args.input_features_paths)
