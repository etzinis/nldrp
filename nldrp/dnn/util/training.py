from __future__ import print_function

import math
import os
import pickle
import sys
import time
import numpy
import torch
from sklearn.utils import compute_class_weight
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader

from nldrp.dnn.config import DNN_BASE_PATH
from nldrp.dnn.logger.experiment import Metric, Experiment
from nldrp.dnn.logger.inspection import Inspector
from nldrp.dnn.util.multi_gpu import get_gpu_id


def sort_batch(lengths):
    """
    Sort batch data and labels by length.
    Useful for variable length inputs, for utilizing PackedSequences
    Args:
        lengths (nn.Tensor): tensor containing the lengths for the data

    Returns:
        - sorted lengths Tensor
        - sort (callable) which will sort a given iterable according to lengths
        - unsort (callable) which will revert a given iterable to its
            original order

    """
    batch_size = lengths.size(0)

    sorted_lengths, sorted_idx = lengths.sort()
    _, original_idx = sorted_idx.sort(0, descending=True)
    reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

    sorted_lengths = sorted_lengths[reverse_idx]

    def sort(iterable):
        if iterable.is_cuda:
            return iterable[sorted_idx.cuda(get_gpu_id())][
                reverse_idx.cuda(get_gpu_id())]
        else:
            return iterable[sorted_idx][reverse_idx]

    def unsort(iterable):
        if iterable.is_cuda:
            return iterable[reverse_idx.cuda(get_gpu_id())][
                original_idx.cuda(get_gpu_id())][
                reverse_idx.cuda(get_gpu_id())]
        else:
            return iterable[reverse_idx][original_idx][reverse_idx]

    return sorted_lengths, sort, unsort


def epoch_progress(loss, epoch, batch, batch_size, dataset_size):
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Batch Loss ({}): {:.4f}'.format(epoch, batch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def get_class_labels(y):
    """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive',
                                    'neutral', 'positive', ...]
    :return: sorted unique class labels
    """
    return numpy.unique(y)


def get_class_weights(y):
    """
    Returns the normalized weights for each class
    based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    weights = compute_class_weight('balanced', numpy.unique(y), y)

    d = {c: w for c, w in zip(numpy.unique(y), weights)}

    return d


def class_weigths(targets, to_pytorch=False):
    w = get_class_weights(targets)
    labels = get_class_labels(targets)
    if to_pytorch:
        return torch.FloatTensor([w[l] for l in sorted(labels)])
    return labels


def _get_predictions(posteriors, task):
    """

    Args:
        posteriors (numpy.array):

    Returns:

    """

    if task == "clf":
        if posteriors.shape[1] > 1:
            predicted = numpy.argmax(posteriors, 1)
        else:
            predicted = numpy.clip(numpy.sign(posteriors), a_min=0,
                                   a_max=None)

    elif task == "multi-clf":
        predicted = numpy.clip(numpy.sign(posteriors), a_min=0,
                               a_max=None)

    elif task == "reg":
        predicted = posteriors

    else:
        raise ValueError

    return predicted


def predict(model, pipeline, dataloader, task,
            mode="eval",
            label_transformer=None):
    """
    Pass a dataset(dataloader) to the model and get the predictions
    Args:
        dataloader (DataLoader): a torch DataLoader which will be used for
            evaluating the performance of the model
        mode (): set the operation mode of the model.
            - "eval" : disable regularization layers
            - "train" : enable regularization layers (MC eval)
        model ():
        pipeline ():
        task ():
        label_transformer ():

    Returns:

    """
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError

    posteriors = []
    y_pred = []
    y = []
    attentions = []
    total_loss = 0

    for i_batch, sample_batched in enumerate(dataloader, 1):
        outputs, labels, atts, loss = pipeline(model, sample_batched)

        if loss is not None:
            total_loss += loss.data[0]

        # get the model posteriors
        posts = outputs.data.cpu().numpy()

        # get the actual predictions (classes and so on...)
        predicted = _get_predictions(posts, task)

        # to numpy
        labels = list(labels.data.cpu().numpy().squeeze())
        predicted = list(predicted.squeeze())

        # make transformations to the predictions
        if label_transformer is not None:
            labels = [label_transformer.inverse(x) for x in labels]
            labels = numpy.array(labels)
            predicted = [label_transformer.inverse(x) for x in predicted]
            predicted = numpy.array(predicted)

        y.extend(labels)
        y_pred.extend(predicted)
        posteriors.extend(list(posts.squeeze()))
        attentions.extend(list(atts.data.cpu().numpy().squeeze()))

    avg_loss = total_loss / i_batch

    return avg_loss, (y, y_pred), posteriors, attentions


def mc_predict(model, pipeline, dataloader, task, label_transformer=None,
               runs=100):
    """
        Monte Carlo predict
    Args:
        model ():
        pipeline ():
        dataloader ():
        task ():
        label_transformer ():
        runs ():

    Returns:

    """
    y = None
    posteriors = []
    avg_losses = []
    for i in range(runs):
        avg_loss, (y, _), _posteriors, attentions = predict(model, pipeline,
                                                            dataloader,
                                                            task, "train",
                                                            label_transformer)
        posteriors.append(_posteriors)
        avg_losses.append(avg_loss)

    # convert to numpy.ndarray in order to utilize scipy's methods
    posteriors = numpy.array(posteriors)

    means = numpy.mean(posteriors, axis=0)
    # stds = numpy.std(posteriors, axis=0)

    predictions = _get_predictions(means, task)

    return numpy.mean(avg_losses), (y, predictions)


class LabelTransformer:
    def __init__(self, map, inv_map=None):
        """
        Class for creating a custom mapping of the labels to ids and back
        Args:
            map (dict):
            inv_map (dict):
        """
        self.map = map
        self.inv_map = inv_map

        if self.inv_map is None:
            self.inv_map = {v: k for k, v in self.map.items()}

    def transform(self, label):
        return self.map[label]

    def inverse(self, label):
        return self.inv_map[label]


class MetricWatcher:
    """
    Base class which monitors a given metric on a Trainer object
    and check whether the model has been improved according to this metric
    """

    def __init__(self, metric, mode="min", base=None):
        self.best = base
        self.metric = metric
        self.mode = mode
        self.scores = None  # will be filled by the Trainer instance

    def has_improved(self):

        # get the latest value for the desired metric
        value = self.scores[self.metric][-1]

        # init best value
        if self.best is None or math.isnan(self.best):
            self.best = value
            return True

        if (
                self.mode == "min" and value < self.best
                or
                self.mode == "max" and value > self.best
        ):  # the performance of the model has been improved :)
            self.best = value
            return True
        else:
            # no improvement :(
            return False


class EarlyStop(MetricWatcher):
    def __init__(self, metric, mode="min", patience=0):
        """

        Args:
            patience (int): for how many epochs to wait, for the performance
                to improve.
            mode (str, optional): Possible values {"min","max"}.
                - "min": save the model if the monitored metric is decreased.
                - "max": save the model if the monitored metric is increased.
        """
        MetricWatcher.__init__(self, metric, mode)
        self.patience = patience
        self.patience_left = patience
        self.best = None

    def stop(self):
        """
        Check whether we should stop the training
        """

        if self.has_improved():
            self.patience_left = self.patience  # reset patience
        else:
            self.patience_left -= 1  # decrease patience

        print(
            "patience left:{}, best({})".format(self.patience_left, self.best))

        # if no more patience left, then stop training
        return self.patience_left < 0


class Checkpoint(MetricWatcher):
    def __init__(self, name, model, metric, model_conf, mode="min",
                 dir=None,
                 base=None,
                 timestamp=False,
                 scorestamp=False,
                 keep_best=False):
        """

        Args:
            model (nn.Module):
            name (str): the name of the model
            mode (str, optional): Possible values {"min","max"}.
                - "min": save the model if the monitored metric is decreased.
                - "max": save the model if the monitored metric is increased.
            keep_best (bool): if True then keep only the best checkpoint
            timestamp (bool): if True add a timestamp to the checkpoint files
            scorestamp (bool): if True add the score to the checkpoint files
            dir (str): the directory in which the checkpoint files will be saved
        """
        MetricWatcher.__init__(self, metric, mode, base)

        self.name = name
        self.dir = dir
        self.model = model
        self.model_conf = model_conf
        self.timestamp = timestamp
        self.scorestamp = scorestamp
        self.keep_best = keep_best
        self.last_saved = None

        if self.dir is None:
            self.dir = os.path.join(DNN_BASE_PATH, "trained")

    def _define_cp_name(self):
        """
        Define the checkpoint name
        Returns:

        """
        fname = [self.name]

        if self.scorestamp:
            score_str = "{:.4f}".format(self.best)
            fname.append(score_str)

        if self.timestamp:
            date_str = time.strftime("%Y-%m-%d_%H:%M")
            fname.append(date_str)

        return "_".join(fname)

    def _save_checkpoint(self):
        """
        A checkpoint saves:
            - the model itself
            - the model's config, which is required for loading related data,
            such the word embeddings, on which it was trained
        Returns:

        """
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        name = self._define_cp_name()
        file_cp = os.path.join(self.dir, name + ".model")
        file_conf = os.path.join(self.dir, name + ".conf")

        # remove previous checkpoint files, if keep_best is True
        if self.keep_best and self.last_saved is not None:
            os.remove(self.last_saved["model"])
            os.remove(self.last_saved["config"])

        # update last saved checkpoint files
        self.last_saved = {
            "model": file_cp,
            "config": file_conf
        }

        # save the checkpoint files (model, model config)
        torch.save(self.model, file_cp)
        with open(file_conf, 'wb') as f:
            pickle.dump(self.model_conf, f)

    def check(self):
        """
        Check whether the model has improved and if so, then save a checkpoint
        Returns:

        """
        if self.has_improved():
            print("Improved model ({}:{:.4f})! "
                  "Saving checkpoint...".format(self.metric, self.best))
            self._save_checkpoint()


class Trainer:
    def __init__(self, model,
                 train_set,
                 optimizer,
                 pipeline,
                 config,
                 train_batch_size=128,
                 eval_batch_size=512,
                 task="clf",
                 use_exp=False,
                 inspect_weights=False,
                 metrics=None,
                 val_set=None,
                 eval_train=True,
                 checkpoint=None,
                 early_stopping=None):
        """
         The Trainer is responsible for training a model.
         It is a stateful object.
         It holds a set of variables that helps us to abstract
         the training process.

        Args:
            use_exp (bool): if True, use the integrated experiment
                manager. In order to utilize the visualizations provided
                by the experiment manager you should:
                    - run `python -m visdom.server` in a terminal.
                    - access visdom by going to http://localhost:8097

                    https://github.com/facebookresearch/visdom#usage

            model (nn.Module): the pytorch model
            train_set (BaseDataset, dict): a
            optimizer ():
            pipeline (callable): a callback function, which defines the training
                pipeline. it must return 3 things (outputs, labels, loss):
                    - outputs: the outputs (predictions) of the model
                    - labels: the gold labels
                    - loss: the loss

            config (): the config instance with the hyperparams of the model
            train_batch_size (int): the batch size that will be used when
                training a model
            eval_batch_size (int): the batch size that will be used when
                evaluating a model
            task (string): you can choose between {"clf", "reg"},
                for classification and regression respectively.
            metrics (dict): a dictionary with the metrics that will be used
                for evaluating the performance of the model.
                - key: string with the name of the metric.
                - value: a callable, with arguments (y, y_hat) tha returns a
                    score.
            val_set (BaseDataset, dict): optional validation dataset
            eval_train (bool): if True, the at the end of each epoch evaluate
                the performance of the model on the training dataset.
            early_stopping (EarlyStop):
            checkpoint (Checkpoint):
        """
        self.use_exp = use_exp
        self.inspect_weights = inspect_weights
        self.model = model
        self.task = task
        self.train_set = train_set
        self.val_set = val_set
        self.config = config
        self.eval_train = eval_train
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.optimizer = optimizer
        self.pipeline = pipeline
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.metrics = {} if metrics is None else metrics

        self.running_loss = 0.0
        self.epoch = 0

        self._init_watched_metrics()

        self.train_loaders, self.val_loader = self._init_dataloaders()

        if use_exp:
            self.experiment = self._init_experiment()
            if self.inspect_weights:
                self.inspector = Inspector(model, ["std", "mean"])

    def _validate_config(self):
        pass

    def _init_watched_metrics(self):
        self.scores = {k: [] for k, v in self.metrics.items()}

        # we need to attach the metrics dictionary
        # on checkpoint and early_stopping objects
        if self.checkpoint is not None:
            self.checkpoint.scores = self.scores
        if self.early_stopping is not None:
            self.early_stopping.scores = self.scores

    def _update_watched_metrics(self):
        pass

    def _init_experiment_tag(self, dataset, tags, tag):
        if isinstance(dataset, dict):
            for _name, _dataset in dataset.items():
                tags.append("{}_{}".format(tag, _name))
        else:
            tags.append(tag)

    def _init_experiment(self):
        """
        init the experiment,
        which will visualize and log the performance of the model
        Returns:

        """

        # 1 - define tags
        tags = []
        if self.eval_train:
            self._init_experiment_tag(self.train_set, tags, "train")
        if self.val_set is not None:
            self._init_experiment_tag(self.val_set, tags, "val")

        # 2 - define experiment
        experiment = Experiment(name=self.config["name"],
                                desc=str(self.model),
                                hparams=self.config)

        # 3 - define metrics
        for name, metric in self.metrics.items():
            experiment.add_metric(Metric(name=name, tags=tags,
                                         vis_type="line"))
            experiment.add_metric(Metric(name="loss", tags=tags,
                                         vis_type="line"))

        return experiment

    def _update_experiment(self, scores, tag):
        pass

    def _init_dataloaders_train(self, dataset, num_workers=4):
        loader = {
            "train": DataLoader(dataset,
                                batch_size=self.train_batch_size,
                                shuffle=True,
                                num_workers=num_workers),
            "val": DataLoader(dataset,
                              batch_size=self.eval_batch_size,
                              num_workers=num_workers),
        }
        return loader

    def _init_dataloaders(self, num_workers=4):

        """
        define different dataloaders, for each dataset and mode
        we use a different dataloader for training and evaluation, in order
        to be able to use different batch sizes, due to a limitation of
        the pytorch's DataLoader class.

        Returns:

        """
        train_loader = None
        val_loader = None

        # train_loader can be:
        # - a dict of (train, val) dataloaders for a single dataset
        # - a dict of (dataset, (train, val)) dataloaders
        #   for a collection of datasets
        if isinstance(self.train_set, dict):
            train_loader = {}
            for name, dataset in self.train_set.items():
                train_loader[name] = self._init_dataloaders_train(dataset,
                                                                  num_workers)
        else:
            train_loader = self._init_dataloaders_train(self.train_set,
                                                        num_workers)

        if self.val_set is not None:
            if isinstance(self.val_set, dict):
                val_loader = {}
                # loop over all validation datasets
                for name, dataset in self.val_set.items():
                    val_loader[name] = DataLoader(
                        dataset,
                        batch_size=self.eval_batch_size,
                        num_workers=num_workers)
            else:
                val_loader = DataLoader(self.val_set,
                                        batch_size=self.eval_batch_size,
                                        num_workers=num_workers)

        return train_loader, val_loader

    def _model_train_loader(self, loader):
        """
        Run a pass of the model on a given dataloader
        Args:
            loader ():

        Returns:

        """
        running_loss = 0.0
        for i_batch, sample_batched in enumerate(loader, 1):
            # 1 - zero the gradients
            self.optimizer.zero_grad()

            # 2 - compute loss using the provided pipeline
            outputs, labels, attentions, loss = self.pipeline(self.model,
                                                              sample_batched)

            # 3 - backward pass: compute gradient wrt model parameters
            loss.backward()

            # just to be sure... clip gradients with norm > N.
            # apply it only if the model has an RNN in it.
            if len([m for m in self.model.modules()
                    if hasattr(m, 'bidirectional')]) > 0:
                clip_grad_norm(self.model.parameters(),
                               self.config["clip_norm"])

            # 4 - update weights
            self.optimizer.step()

            running_loss += loss.data[0]

            # print statistics
            epoch_progress(loss=loss.data[0],
                           epoch=self.epoch,
                           batch=i_batch,
                           batch_size=loader.batch_size,
                           dataset_size=len(self.train_set))
        return running_loss

    def model_train(self):
        """
        Train the model for one epoch (on one or more dataloaders)
        Returns:

        """
        # switch to train mode -> enable regularization layers, such as Dropout
        self.model.train()
        self.epoch += 1
        running_loss = 0.0

        if isinstance(self.train_set, dict):
            for name, loader in self.train_loaders.items():
                running_loss += self._model_train_loader(loader["train"])
        else:
            running_loss += self._model_train_loader(
                self.train_loaders["train"])

        return running_loss

    def _calc_scores(self, y, y_pred):
        return {name: metric(y, y_pred)
                for name, metric in self.metrics.items()}

    def _model_eval_loader(self, loader, tag):
        """
        Evaluate a dataloader
        and update the corresponding scores and metrics
        Args:
            loader ():
            tag ():

        Returns:

        """
        # 1 - evaluate the dataloader
        label_transformer = self._infer_label_transformer()
        avg_loss, (y, y_pred), posteriors, attentions = predict(
            self.model,
            self.pipeline,
            loader,
            self.task,
            "eval",
            label_transformer)

        # 2 - calculate its performance according to each metric
        scores = self._calc_scores(y, y_pred)
        # 3 - print the scores
        self._print_scores(scores, avg_loss, tag.upper())

        # TEST Monte Carlo Evaluation
        # if tag == "val":
        #     mc_avg_loss, (mc_y, mc_y_pred) = self.mc_predict(loader)
        #     mc_scores = self._calc_scores(mc_y, mc_y_pred)
        #     self.print_scores(mc_scores, mc_avg_loss, "MC_" + tag.upper())

        # 4 - update the corresponding values in the experiment
        if self.use_exp:
            for score, value in scores.items():
                self.experiment.metrics[score].append(tag, value)
            self.experiment.metrics["loss"].append(tag, avg_loss)
        return avg_loss, scores

    def _aggregate_scores(self, scores, aggregate):
        aggs = {k: [score[k] for score in scores]
                for k in scores[0].keys()}
        aggs = {k: aggregate(v) for k, v in aggs.items()}
        return aggs

    def model_eval(self):
        """
        Evaluate the model on each dataset and update the corresponding metrics.
        The function is normally called at the end of each epoch.
        Returns:

        """

        # 1 - evaluate on train datasets
        if self.eval_train:
            if isinstance(self.train_set, dict):
                for name, loader in self.train_loaders.items():
                    tag = "train_{}".format(name)
                    self._model_eval_loader(loader["val"], tag)

            else:
                self._model_eval_loader(self.train_loaders["val"], "train")

        # 2 - evaluate on validation datasets
        if self.val_loader is not None:
            if isinstance(self.val_set, dict):
                loss = []
                scores = []
                for name, loader in self.val_loader.items():
                    tag = "val_{}".format(name)
                    _loss, _scores = self._model_eval_loader(loader, tag)
                    loss.append(_loss)
                    scores.append(_scores)

                agg_scores = self._aggregate_scores(scores, numpy.mean)
                for name, value in agg_scores.items():
                    self.scores[name].append(value)
            else:
                loss, scores = self._model_eval_loader(self.val_loader, "val")
                for name, value in scores.items():
                    self.scores[name].append(value)

        if self.use_exp:
            self.experiment.update_plots()
            if self.inspect_weights:
                self.inspector.update_state(self.model)

    def _print_scores(self, scores, loss, tag):
        """
        Log the scores of a dataset (tag) on the console
        Args:
            scores (): a dictionary of (metric_name, value)
            loss (): the loss of the model on an epoch
            tag (): the dataset (name)

        Returns:

        """
        print("\t{:6s} - ".format(tag), end=" ")
        for name, value in scores.items():
            print(name, '{:.4f}'.format(value), end=", ")
        print(" Loss:{:.4f}".format(loss))

    def _infer_label_transformer(self):
        # both datasets (train,val) should have the same transformer
        if isinstance(self.train_set, dict):
            # pick any dataset from the dict.
            # All should have the same transformer
            dataset = self.train_set[list(self.train_set.keys())[0]]
            return dataset.label_transformer
        else:
            return self.train_set.label_transformer
