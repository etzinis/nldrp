"""!
\brief Utterance Level Model Optimizer

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import copy
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import Optimizer

def generate_speaker_dependent_folds(features_dic,
                                     n_splits=5,
                                     random_seed=7):
    norm_per_sp_dic = copy.deepcopy(features_dic)
    # del norm_per_sp_dic['KL']
    for sp, data in norm_per_sp_dic.items():
        # this_scaler = StandardScaler().fit(data['x'])
        # norm_per_sp_dic[sp]['x'] = this_scaler.transform(data['x'])
        m_vec = np.mean(data['x'], axis=0)
        s_vec = np.std(data['x'], axis=0)
        this_ar = norm_per_sp_dic[sp]['x']
        normed_ar = (this_ar - m_vec)
        with np.errstate(divide='ignore'):
            normed_ar = normed_ar / (s_vec + 1e-6)
        norm_per_sp_dic[sp]['x'] = normed_ar

    xy_l = [v for (sp, v) in norm_per_sp_dic.items()]
    x_all = np.concatenate([v['x'] for v in xy_l])
    y_all = [utt_label for speaker_labels in [v['y'] for v in xy_l]
             for utt_label in speaker_labels]

    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=random_seed)
    for tr_ind, te_ind in skf.split(x_all, y_all):
        yield (x_all[te_ind],
               [y_all[i] for i in te_ind],
               x_all[tr_ind],
               [y_all[i] for i in tr_ind])


def generate_speaker_independent_folds(features_dic):
    ind_dic = copy.deepcopy(features_dic)
    # del ind_dic['KL']

    for te_speaker, te_data in ind_dic.items():
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in ind_dic.items():
            if tr_speaker == te_speaker:
                continue
            x_tr_list.append(tr_data['x'])
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)

        m_vec = np.mean(X_tr, axis=0)
        s_vec = np.std(X_tr, axis=0)
        X_tr_n = (X_tr - m_vec)
        with np.errstate(divide='ignore'):
            X_tr_n = X_tr_n / (s_vec + 1e-6)

        x_te = te_data['x']
        x_te_n = (x_te - m_vec)
        with np.errstate(divide='ignore'):
            x_te_n = x_te_n / (s_vec + 1e-6)

        yield x_te_n, te_data['y'], X_tr_n, Y_tr



def loso_with_best_models(features_dic):
    """!
    \brief This is the function you should call if you have a
    dictionary with keys as the speakers and each one has a
    corresponding 2D matrix and a 1d label vector for each one.
    Namely: converted_dic[spkr]['x'] = X_2D
            converted_dic[spkr]['y'] = y_list"""

    best_models = {}


    # svm_opt_obj = Optimizer.ModelOptimizer(
    #               'svm',
    #               generate_speaker_dependent_folds(features_dic),
    #               {'C': [0.1, 0.3, 0.5, 1, 3, 5, 7, 8, 10],
    #                'kernel': ['rbf']},
    #               ['w_acc', 'uw_acc'])
    #
    # best_models["SVM Dependent"] = svm_opt_obj.optimize_model()

    svm_opt_obj = Optimizer.ModelOptimizer(
        'svm',
        generate_speaker_independent_folds(features_dic),
        {'C': [0.1, 0.3, 0.5, 1, 3, 5, 7, 8, 10],
         'kernel': ['rbf']},
        ['w_acc', 'uw_acc'])

    best_models["SVM Independent"] = svm_opt_obj.optimize_model()

    lr_opt_obj = Optimizer.ModelOptimizer(
        'lr',
        generate_speaker_dependent_folds(features_dic),
        {'C': [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3],
         'penalty': ['l2']},
        ['w_acc', 'uw_acc'])

    best_models["LR Dependent"] = lr_opt_obj.optimize_model()

    lr_opt_obj = Optimizer.ModelOptimizer(
        'lr',
        generate_speaker_independent_folds(features_dic),
        {'C': [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3],
         'penalty': ['l2']},
        ['w_acc', 'uw_acc'])

    best_models["LR Independent"] = lr_opt_obj.optimize_model()

    return best_models


def command_line_optimizer(list_of_paths):

    speakers_dic = fuse_all_configurations(list_of_paths)

    best_models = loso_with_best_models(speakers_dic)

    # print "List in reverse"
    # speakers_dic = fuse_all_configurations(list_of_paths[::-1])
    #
    # best_models = loso_with_best_models(speakers_dic)

    # print "Optimized LR and SVM for both Speaker Dependent and " \
    #       "Independent Experimentations"
    # from pprint import pprint
    # pprint(best_models)


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

        this_utt_array = np.array(x_list, dtype=np.float32)
        converted_dic[spkr]['x'] = this_utt_array
        converted_dic[spkr]['y'] = y_list

    return converted_dic


def fuse_all_configurations(list_of_paths):


    try:
        list_of_dics = [joblib.load(p) for p in list_of_paths]
    except Exception as e:
        print "At least one file path is required"
        raise e

    final_data_dic = list_of_dics.pop(0)

    while list_of_dics:
        temp_dic = list_of_dics.pop(0)
        try:
            for spkr in temp_dic:
                for id, el_dic in temp_dic[spkr].items():
                    assert el_dic['y'] == final_data_dic[spkr][id]['y']
                    prev_vec = list(final_data_dic[spkr][id]['x'])
                    this_vec = list(el_dic['x'])
                    new_vec = np.array(prev_vec+this_vec,
                                       dtype=np.float32)

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
