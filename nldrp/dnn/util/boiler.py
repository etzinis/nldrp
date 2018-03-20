"""
This file contains functions with logic that is used in almost all models,
with the goal of avoiding boilerplate code (and bugs due to copy-paste),
such as training pipelines.
"""
from __future__ import print_function

import glob
import os
import pickle

import torch
from torch.autograd import Variable

from nldrp.dnn.config import DNN_BASE_PATH
from nldrp.dnn.util.multi_gpu import get_gpu_id
from nldrp.dnn.util.training import sort_batch


def load_pretrained_model(name):
    model_path = os.path.join(DNN_BASE_PATH, "trained",
                              "{}.model".format(name))
    model_conf_path = os.path.join(DNN_BASE_PATH, "trained",
                                   "{}.conf".format(name))
    model = torch.load(model_path)
    model_conf = pickle.load(open(model_conf_path, 'rb'))

    return model, model_conf


def load_pretrained_models(name):
    models_path = os.path.join(DNN_BASE_PATH, "trained")
    fmodel_confs = sorted(glob.glob(os.path.join(models_path,
                                                 "{}*.conf".format(name))))
    fmodels = sorted(glob.glob(os.path.join(models_path,
                                            "{}*.model".format(name))))
    for model, model_conf in zip(fmodels, fmodel_confs):
        print("loading model {}".format(model))
        yield torch.load(model), pickle.load(open(model_conf, 'rb'))


def pipeline_classification(criterion=None, eval=False):
    """
    Generic classification pipeline
    Args:
        criterion (): the loss function
        binary (bool): set to True for binary classification
        eval (): set to True if the pipeline will be used
            for evaluation and not for training.
            Note: this has nothing to do with the mode
            of the model (eval or train). If the pipeline will be used
            for making predictions, then set to True.

    Returns:

    """

    def pipeline(nn_model, batch):
        # convert to Variables
        batch = map(lambda x: Variable(x), batch)

        # convert to CUDA
        if torch.cuda.is_available():
            batch = map(lambda x: x.cuda(get_gpu_id()), batch)

        inputs, labels, lengths = batch

        outputs, attentions = nn_model(inputs, lengths)

        if eval:
            return outputs, labels, attentions, None

        loss = criterion(outputs, labels)

        return outputs, labels, attentions, loss

    return pipeline
