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
from sklearn.preprocessing import StandardScaler


def generate_speaker_dependent_folds(features_dic):
    for te_speaker, te_data in features_dic.items():
        scaler = StandardScaler().fit(te_data['x'])
        x_te = scaler.transform(te_data['x'])
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in features_dic.items():
            if tr_speaker == te_speaker:
                continue
            scaler = StandardScaler().fit(tr_data['x'])
            sp_x = scaler.transform(tr_data['x'])
            x_tr_list.append(sp_x)
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        yield te_speaker, x_te, te_data['y'], X_tr, Y_tr


def generate_speaker_independent_folds(features_dic):
    all_X = np.concatenate([v['x'] for k, v in features_dic.items()],
                           axis=0)
    all_scaler = StandardScaler().fit(all_X)

    for te_speaker, te_data in features_dic.items():
        x_te = all_scaler.transform(te_data['x'])
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in features_dic.items():
            if tr_speaker == te_speaker:
                continue
            sp_x = all_scaler.transform(tr_data['x'])
            x_tr_list.append(sp_x)
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        yield te_speaker, x_te, te_data['y'], X_tr, Y_tr



def loso_with_best_models(features_dic):
    """!
    \brief This is the function you should call if you have a
    dictionary with keys as the speakers and each one has a
    corresponding 2D matrix and a 1d label vector for each one.
    Namely: converted_dic[spkr]['x'] = X_2D
            converted_dic[spkr]['y'] = y_list"""

    for te_speaker, X_te, Y_te, X_tr, Y_tr in \
            generate_speaker_dependent_folds(features_dic):
        print X_tr[0, 4:8]

    for te_speaker, X_te, Y_te, X_tr, Y_tr in \
            generate_speaker_independent_folds(features_dic):
        print X_tr[0, 4:8]




def command_line_optimizer(list_of_paths):

    speakers_dic = fuse_all_configurations(list_of_paths)

    loso_with_best_models(speakers_dic)


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
        converted_dic[spkr]['x'] = this_utt_array
        converted_dic[spkr]['y'] = y_list

    return converted_dic


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
    return fused_converted_dic

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
    command_line_optimizer(args.input_features_paths)
