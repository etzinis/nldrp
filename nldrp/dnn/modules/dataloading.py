import os
import pickle

import numpy

from torch.utils.data import Dataset

from nldrp.dnn.config import DNN_BASE_PATH


class BaseDataset(Dataset):
    """
    This is a Base class which extends pytorch's Dataset, in order to avoid
    boilerplate code and equip our datasets with functionalities such as
    caching.
    """

    def __init__(self, X, y,
                 max_length=0,
                 name=None,
                 label_transformer=None):
        """

        Args:
            X (): List of training samples
            y (): List of training labels
            name (str): the name of the dataset. It is needed for caching.
                if None then caching is disabled.
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            label_transformer (LabelTransformer):
        """
        self.data = X
        self.labels = y
        self.name = name
        self.label_transformer = label_transformer

        self.data = self.load_data()

        self.set_max_length(max_length)

    def set_max_length(self, max_length):
        # if max_length == 0, then set max_length
        # to the maximum sentence length in the dataset
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(DNN_BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self):
        return os.path.join(DNN_BASE_PATH, "_cache",
                            "preprocessed_{}.p".format(self.name))

    def _write_cache(self, data):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load_data(self):

        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.preprocess(self.name, self.data)

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(self.name, self.data)
            self._write_cache(data)
            return data


class EmotionDataset(Dataset):

    def __init__(self, data, targets, data_manager, transform=None):
        """

        Args:
            transform (list): a list of callable that apply transformation
                on the samples.
        """

        if transform is None:
            transform = []
        self.transform = transform

        self.data = data
        self.targets = targets
        self.data_mngr = data_manager

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
            * example (ndarray): vector representation of a training example
            * label (string): the class label
            * length (int): the length (segments) of the utterance
            * index (int): the index of the returned dataitem in the dataset.
              It is useful for getting the raw input for visualizations.

        """
        sample = self.data[index]
        label = self.targets[0][index], self.targets[1][index]

        for i, tsfrm in enumerate(self.transform):
            sample = tsfrm(sample)

        # standardize the data
        sample = self.data_mngr.normalizer.transform(sample).astype('float32')
        # zero padding, up to self.max_length
        sample = self.data_mngr.pad_sample(sample).astype('float32')

        # convert string categorical labels, to class ids
        cat_label = self.data_mngr.label_cat_encoder.transform([label[1]])[0]
        # convert continuous labels, to desired range (0-1)
        cont_label = self.data_mngr.label_cont_encoder.transform(
            [label[0]]).ravel().astype('float32')

        return sample, cont_label, cat_label, len(self.data[index]), index
