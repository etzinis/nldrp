"""!
\brief Optimizer class wrapper for abstracting the functionality of
training -- evaluating models -- providing the best model according
to the specification for a metric.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""


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
                 params_grid_space,
                 metric_to_optimize):



