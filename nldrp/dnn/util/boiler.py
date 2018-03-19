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
from nldrp.config import BASE_PATH
from nldrp.dnn.util.training import sort_batch


def load_pretrained_model(name):
    model_path = os.path.join(BASE_PATH, "semeval2018/trained",
                              "{}.model".format(name))
    model_conf_path = os.path.join(BASE_PATH, "semeval2018/trained",
                                   "{}.conf".format(name))
    model = torch.load(model_path)
    model_conf = pickle.load(open(model_conf_path, 'rb'))

    return model, model_conf


def load_pretrained_models(name):
    models_path = os.path.join(BASE_PATH, "semeval2018/trained")
    fmodel_confs = sorted(glob.glob(os.path.join(models_path,
                                                 "{}*.conf".format(name))))
    fmodels = sorted(glob.glob(os.path.join(models_path,
                                            "{}*.model".format(name))))
    for model, model_conf in zip(fmodels, fmodel_confs):
        print("loading model {}".format(model))
        yield torch.load(model), pickle.load(open(model_conf, 'rb'))


def load_datasets(X_train, y_train, X_test, y_test, op_mode, params=None,
                  word2idx=None, label_transformer=None, emojis=False,
                  test_mode=False):
    if params is not None:
        name = "_".join(params) if isinstance(params, list) else params
    else:
        name = None

    train_set = None
    val_set = None

    if op_mode == "word":
        if word2idx is None:
            raise ValueError

        preprocessor = twitter_preprocess()

        print("Building word-level datasets...")
        train_set = WordDataset(
            X_train, y_train, word2idx,
            name="{}_train".format(name) if name is not None else None,
            preprocess=preprocessor,
            label_transformer=label_transformer)
        if not test_mode:
            val_set = WordDataset(
                X_test, y_test, word2idx,
                name="{}_val".format(name) if name is not None else None,
                preprocess=preprocessor,
                label_transformer=label_transformer)
    elif op_mode == "char":
        print("Building char-level datasets...")
        train_set = CharDataset(
            X_train, y_train,
            name="{}_char_train".format(name) if name is not None else None,
            emojis=emojis,
            label_transformer=label_transformer)
        if not test_mode:
            val_set = CharDataset(
                X_test, y_test,
                name="{}_char_val".format(name) if name is not None else None,
                emojis=emojis,
                label_transformer=label_transformer)

    return train_set, val_set


def load_embeddings(model_conf):
    word_vectors = os.path.join(BASE_PATH, "embeddings",
                                "{}.txt".format(model_conf["embeddings_file"]))
    word_vectors_size = model_conf["embeddings_dim"]

    # load word embeddings
    print("loading word embeddings...")
    return load_word_vectors(word_vectors, word_vectors_size)


def pipeline_classification(criterion=None, binary=False, eval=False):
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

    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        inputs, labels, lengths, indices = curr_batch

        # sort batch (for handling inputs of variable length)
        lengths, sort, unsort = sort_batch(lengths)

        # sort Variables
        inputs = sort(inputs)
        labels = sort(labels)

        inputs = Variable(inputs)
        labels = Variable(labels)
        lengths = Variable(lengths)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            lengths = lengths.cuda()

        outputs, attentions = nn_model(inputs, lengths)

        # unsort Variables (preserve original order)
        outputs = unsort(outputs)
        if attentions is not None:
            attentions = unsort(attentions)
        labels = unsort(labels)

        if eval:
            return outputs, labels, attentions, None

        if binary:
            loss = criterion(outputs.view(-1), labels.float())
        else:
            loss = criterion(outputs, labels)

        return outputs, labels, attentions, loss

    return pipeline


def pipeline_classification_tagged(criterion=None, eval=False):
    """
    Classification pipeline, with tag, for multi-task learning
    Args:
        criterion (): the loss function
        eval (): set to True if the pipeline will be used
            for evaluation and not for training.
            Note: this has nothing to do with the mode
            of the model (eval or train). If the pipeline will be used
            for making predictions, then set to True.
    Returns:

    """

    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        inputs, labels, tag, lengths, indices = curr_batch

        # sort batch (for handling inputs of variable length)
        lengths, sort, unsort = sort_batch(lengths)

        # sort Variables
        inputs = sort(inputs)
        labels = sort(labels)
        tag = sort(tag)

        inputs = Variable(inputs)
        labels = Variable(labels)
        tag = Variable(tag)
        lengths = Variable(lengths)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            tag = tag.cuda()
            lengths = lengths.cuda()

        outputs, attentions = nn_model(inputs, tag, lengths)

        # unsort Variables (preserve original order)
        outputs = unsort(outputs)
        attentions = unsort(attentions)
        labels = unsort(labels)

        if eval:
            return outputs, labels, attentions, None

        loss = criterion(outputs, labels)

        return outputs, labels, attentions, loss

    return pipeline


def pipeline_regression(criterion=None, eval=False):
    """
    Suitable for regression, or multi-label classification (binary for each yi)
    Args:
        criterion (): the loss function
        eval (): set to True if the pipeline will be used
            for evaluation and not for training.
            Note: this has nothing to do with the mode
            of the model (eval or train). If the pipeline will be used
            for making predictions, then set to True.

    Returns:

    """

    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        inputs, labels, lengths, indices = curr_batch

        # sort batch (for handling inputs of variable length)
        lengths, sort, unsort = sort_batch(lengths)

        # sort Variables
        inputs = sort(inputs)
        labels = sort(labels)

        # convert to Variables
        inputs = Variable(inputs)
        labels = Variable(labels.float())
        lengths = Variable(lengths)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            lengths = lengths.cuda()

        outputs, attentions = nn_model(inputs, lengths)

        # unsort Variables (preserve original order)
        outputs = unsort(outputs)
        attentions = unsort(attentions)
        labels = unsort(labels)

        if eval:
            return outputs, labels, attentions, None

        loss = criterion(outputs, labels)

        return outputs, labels, attentions, loss

    return pipeline


def pipeline_regression_tagged(criterion=None, eval=False):
    """
    Generic pipeline for regression, but with support for multi-task learning
    with the addition the task-specific tag
    Args:
        criterion (): the loss function
        eval (): set to True if the pipeline will be used
            for evaluation and not for training.
            Note: this has nothing to do with the mode
            of the model (eval or train). If the pipeline will be used
            for making predictions, then set to True.

    Returns:

    """

    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        inputs, labels, tag, lengths, indices = curr_batch

        # sort batch (for handling inputs of variable length)
        lengths, sort, unsort = sort_batch(lengths)

        # sort Variables
        inputs = sort(inputs)
        labels = sort(labels)
        tag = sort(tag)

        inputs = Variable(inputs)
        labels = Variable(labels.float())
        tag = Variable(tag)
        lengths = Variable(lengths)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            tag = tag.cuda()
            lengths = lengths.cuda()

        outputs, attentions = nn_model(inputs, tag, lengths)

        # unsort Variables (preserve original order)
        outputs = unsort(outputs)
        attentions = unsort(attentions)
        labels = unsort(labels)

        if eval:
            return outputs, labels, attentions, None

        loss = criterion(outputs, labels)

        return outputs, labels, attentions, loss

    return pipeline
