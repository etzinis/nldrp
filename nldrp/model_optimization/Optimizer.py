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
import itertools


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


def dummy_generate_SVMs_and_LRs():
    svm_params = [('svm', c) for c in [0.1, 0.3, 0.5, 1, 3, 5, 7, 8,
                                       10]]
    lr_params = [('lr', c) for c in [0.1, 0.3, 0.5, 1, 3, 5, 7, 8, 10]]
    all_params = svm_params + lr_params

    for m_name, c in all_params:
        if m_name == 'svm':
            yield SVC(C=c)
        else:
            yield LogisticRegression(C=c)


class ModelOptimizer(object):
    def __init__(self,
                 model_name,
                 folds_generator,
                 params_grid,
                 metrics_to_optimize):

        valid_models = ['svm', 'lr']
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
    def configure_model(model_name, params):
        if model_name == 'svm':
            model = SVC(C=params.get('C', 1),
                        kernel=params.get('kernel', 'rbf'))
        elif model_name == 'lr':
            model = LogisticRegression(C=params.get('C', 1),
                                       penalty=params.get('penalty',
                                                          'l2'))
        else:
            valid_models = ['svm', 'lr']
            raise NotImplementedError(('Model: <{}> is not yet '
                                       'supported. Please try one of '
                                       'the following: {}'.format(
                model_name, valid_models)))

        return model

    def generate_grid_space(self):
        keys, values = zip(*self.param_grid.items())
        experiments = [dict(zip(keys, v)) for v in
                       itertools.product(*values)]
        fold_gens = itertools.tee(self.folds_gen, len(experiments))
        for i, v in enumerate(experiments):
            yield v, fold_gens[i]

    def evaluate_model(self, model, folds_gen):
        model_metrics = {}
        for x_te, y_te, x_tr, y_tr in folds_gen:
            model.fit(x_tr, y_tr)
            y_pred = model.predict(x_te)
            this_f_metrics = compute_metrics(y_pred, y_te)
            for m in this_f_metrics:
                if m not in model_metrics:
                    model_metrics[m] = [this_f_metrics[m]]
                else:
                    model_metrics[m].append(this_f_metrics[m])

        for m in model_metrics:
            model_metrics[m] = np.mean(model_metrics[m])
        return model_metrics


    def optimize_model(self):
        grid_space = self.generate_grid_space()
        for config_params, fold_gen in grid_space:
            model = self.configure_model(self.model_name,
                                         config_params)
            metrics = self.evaluate_model(model, fold_gen)
            print config_params
            print metrics


if __name__ == "__main__":
    for model in dummy_generate_SVMs_and_LRs():
        print model
