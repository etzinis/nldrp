"""!
\brief Optimizer class wrapper for abstracting the functionality of
training -- evaluating models -- providing the best model according
to the specification for a metric.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



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


class ModelOptimizer(object):
    def __init__(self,
                 model_name,
                 folds_generator,
                 params_grid,
                 metrics_to_optimize):

        valid_models = ['SVM', 'LR']
        if model_name in valid_models:
            self.model_name = model_name
        else:
            raise NotImplementedError(('Model: <{}> is not yet '
                                       'supported. Please try one of '
                                       'the following: {}'.format(
                                        model_name, valid_models)))

        self.param_grid = params_grid
        self.folds_gen = folds_generator

        valid_metrics = ['uw_f1', 'uw_acc', 'w_acc']
        assert type(metrics_to_optimize) == type([]), 'Metrics to ' \
                                                      'optimize ' \
                                                      'should be ' \
                                                      'defined in a ' \
                                                      'list'
        for m in metrics_to_optimize:
            if m not in valid_metrics:
                raise NotImplementedError(('Metric to optimize: <{}> '
                                           'is not yet '
                                           'supported. Please try '
                                           'one of '
                                           'the following: {}'.format(
                                            m, valid_metrics)))
        self.opt_metrics = metrics_to_optimize


    @staticmethod
    def configure_model(model_name):
        if model_name == 'SVM':
            return







