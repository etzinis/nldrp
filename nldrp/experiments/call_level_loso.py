"""!
\brief Utterance level classification schema by utilizing various
models for the configured experiment.

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
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('ADAb', AdaBoostClassifier()))
    models.append(('GRADb', GradientBoostingClassifier()))
    models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('LinR', LogisticRegression()))
    return dict(models)


def evaluate_fold(model,
                  X_te, Y_te,
                  X_tr, Y_tr):

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import LocallyLinearEmbedding
    # pca = PCA(n_components=int(X_tr.shape[1] / 10)).fit(X_tr)
    n_components = int(X_tr.shape[1] / 10)
    pca = LocallyLinearEmbedding(n_components=n_components,
                                 n_neighbors=(n_components + 1),
                                 method='modified').fit(X_tr)
    X_tr = pca.transform(X_tr)
    scaler = StandardScaler().fit(X_tr)
    X_tr_scaled = scaler.transform(X_tr)
    model.fit(X_tr_scaled, Y_tr)
    X_te = pca.transform(X_te)
    X_te_scaled = scaler.transform(X_te)
    Y_pred = model.predict(X_te_scaled)
    model_metrics = compute_metrics(Y_pred, Y_te)
    return model_metrics

def loso(fusion_method, config):

    features_dic = feature_loader.load_and_convert(fusion_method, config)
    all_models = configure_models()
    result_dic = {}

    for model_name, model in all_models.items():
        result_dic[model_name] = {}
        for te_speaker, X_te, Y_te, X_tr, Y_tr in generate_speaker_folds(
            features_dic):

            fold_info = evaluate_fold(model, X_te, Y_te, X_tr, Y_tr)
            print model_name, te_speaker
            pprint.pprint(fold_info)
            if result_dic[model_name]:
                for k,v in fold_info.items():
                    result_dic[model_name][k].append(v)
            else:
                for k, v in fold_info.items():
                    result_dic[model_name][k] = [v]

        for k, v in result_dic[model_name].items():
            result_dic[model_name][k] = '{} +- {}'.format(
                np.mean(v), np.std(v))


    # pprint.pprint(result_dic)



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
