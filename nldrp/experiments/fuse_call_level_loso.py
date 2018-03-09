"""!
\brief Utterance level classification schema by concatenating vectors
and performing classification

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.externals import joblib

import argparse
import numpy as np
import pprint
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
        yield te_speaker, te_data['x'], te_data['y'], X_tr, Y_tr


def compute_metrics(Y_predicted, Y_true):
    uw_f1 = f1_score(Y_predicted, Y_true, average='macro')
    w_f1 = f1_score(Y_predicted, Y_true, average='micro')

    uw_rec = recall_score(Y_predicted, Y_true, average='macro')
    w_rec = recall_score(Y_predicted, Y_true, average='micro')

    uw_prec = precision_score(Y_predicted, Y_true, average='macro')
    w_prec = precision_score(Y_predicted, Y_true, average='micro')

    w_acc = accuracy_score(Y_predicted, Y_true)
    cmat = confusion_matrix(Y_predicted, Y_true)
    with np.errstate(divide='ignore'):
        uw_acc = (cmat.diagonal() / (1.0 * cmat.sum(axis=1) + 1e-6
                                     )).mean()
        if np.isnan(uw_acc):
            uw_acc = 0.

    metrics_l = [('uw_f1', uw_f1),
                 ('w_f1', w_f1),
                 ('uw_rec', uw_rec),
                 ('w_rec', w_rec),
                 ('uw_prec', uw_prec),
                 ('w_prec', w_prec),
                 ('uw_acc', uw_acc),
                 ('w_acc', w_acc)]

    metric_dic = dict(metrics_l)
    return metric_dic


def configure_models():
    models = []
    # models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier()))
    # models.append(('ADAb', AdaBoostClassifier()))
    # models.append(('GRADb', GradientBoostingClassifier()))
    # models.append(('QDA', QuadraticDiscriminantAnalysis()))
    # models.append(('LinR', LogisticRegression()))
    return dict(models)


def evaluate_fold(model,
                  X_te, Y_te,
                  X_tr, Y_tr):

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import LocallyLinearEmbedding

    n_components = int(X_tr.shape[1] / 2)
    # n_components = 1500
    pca = PCA(n_components=n_components).fit(X_tr)
    # # pca = LocallyLinearEmbedding(n_components=n_components,
    # #                              n_neighbors=(n_components + 1),
    # #                              method='modified').fit(X_tr)
    # X_tr = pca.transform(X_tr)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    model.fit(X_tr, Y_tr)
    # X_te = pca.transform(X_te)
    scaler_te = StandardScaler().fit(X_te)
    X_te = scaler_te.transform(X_te)
    Y_pred = model.predict(X_te)
    model_metrics = compute_metrics(Y_pred, Y_te)
    return model_metrics


def evaluate_loso(features_dic):
    all_models = configure_models()
    result_dic = {}

    for model_name, model in all_models.items():
        result_dic[model_name] = {}
        for te_speaker, X_te, Y_te, X_tr, Y_tr in generate_speaker_folds(
                features_dic):

            fold_info = evaluate_fold(model, X_te, Y_te, X_tr, Y_tr)

            if result_dic[model_name]:
                for k, v in fold_info.items():
                    result_dic[model_name][k].append(v)
            else:
                for k, v in fold_info.items():
                    result_dic[model_name][k] = [v]

        print model_name
        for k, v in result_dic[model_name].items():
            result_dic[model_name][k] = '{} +- {}'.format(
                np.mean(v), np.std(v))
            print k, result_dic[model_name][k]

    # pprint.pprint(result_dic)


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


def fusion_loso(list_of_paths):

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
    evaluate_loso(fused_converted_dic)


def get_args():
    """! Command line parser for Utterance level classification Leave
    one speaker out schema pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level classification Leave one '
                    'speaker out schema pipeline' )
    parser.add_argument('-i', '--input_features_paths', nargs='+',
                        help='File paths of the features you want to '
                             'concatenate and the classify')

    # parser.add_argument("--dataset", type=str,
    #                     help="""The name of the dataset""",
    #                     required=True,
    #                     choices=['SAVEE'])
    # parser.add_argument("-i", "--save_dir", type=str,
    #     help="""Where the corresponding binary file full of
    #     data that will contain the dictionary for each speaker is
    #     stored.
    #     Another subdic for all the sentences with their ids
    #     and a 1d numpy matrix for each one of them.
    #     """,
    #     default=nldrp.config.EXTRACTED_FEATURES_PATH )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    fusion_loso(args.input_features_paths)