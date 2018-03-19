import numpy
from torch.utils.data import Dataset


class EmotionDataset(Dataset):

    def __init__(self, data, labels,
                 data_transformer=None,
                 label_transformer=None,
                 max_length=None):

        self.data = data
        self.labels = labels
        self.data_transformer = data_transformer
        self.label_transformer = label_transformer

        if max_length is None:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def pad_sample(self, sample):
        z_pad = numpy.zeros((self.max_length, sample.shape[1]))
        z_pad[:sample.shape[0], :sample.shape[1]] = sample
        return z_pad

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
        label = self.labels[index]

        if self.data_transformer is not None:
            sample = self.data_transformer.transform(sample).astype('float32')

        if self.label_transformer is not None:
            label = self.label_transformer.map[label]

        # zero padding, up to self.max_length
        sample = self.pad_sample(sample).astype('float32')

        return sample, label, len(self.data[index])
