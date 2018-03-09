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


def fusion_loso(list_of_paths):
    pass


