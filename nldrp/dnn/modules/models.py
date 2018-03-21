import torch
from torch import nn
from torch.autograd import Variable

from nldrp.dnn.modules.attention import SelfAttention
from nldrp.dnn.modules.modules import RecurrentEncoder
from nldrp.dnn.modules.regularization import GaussianNoise
from nldrp.dnn.util.multi_gpu import get_gpu_id


class ModelHelper:
    def _sort_by(self, lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda(get_gpu_id())

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):
            if len(iterable.shape) > 1:
                return iterable[sorted_idx.data][reverse_idx]
            else:
                return iterable

        def unsort(iterable):
            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort

    def _get_mask(self, sequence, lengths, axis=0):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(sequence.size(axis))).detach()

        if sequence.data.is_cuda:
            mask = mask.cuda(get_gpu_id())

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask


class EmotionModel(nn.Module, ModelHelper):
    def __init__(self, input_size, classes, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            out_size ():
        """
        super(EmotionModel, self).__init__()

        input_noise = kwargs.get("input_noise", 0.)
        input_dropout = kwargs.get("input_dropout", 0.2)

        encoder_size = kwargs.get("encoder_size", 128)
        encoder_layers = kwargs.get("encoder_layers", 1)
        encoder_dropout = kwargs.get("encoder_dropout", 0.2)

        attention_layers = kwargs.get("attention_layers", 1)
        attention_dropout = kwargs.get("attention_dropout", 0.)
        attention_activation = kwargs.get("attention_activation", "tanh")

        encoder_type = kwargs.get("encoder_type", "LSTM")
        bidirectional = kwargs.get("bidirectional", False)

        ######################################################################
        # Layers
        ######################################################################
        self.drop_input = nn.Dropout(input_dropout)
        self.noise_input = GaussianNoise(input_noise)

        self.encoder = RecurrentEncoder(input_size=input_size,
                                        rnn_size=encoder_size,
                                        rnn_type=encoder_type,
                                        num_layers=encoder_layers,
                                        bidirectional=bidirectional,
                                        dropout=encoder_dropout)

        feature_size = encoder_size
        if bidirectional:
            feature_size *= 2

        self.attention = SelfAttention(feature_size,
                                       layers=attention_layers,
                                       dropout=attention_dropout,
                                       non_linearity=attention_activation,
                                       batch_first=True)

        self.classifier = nn.Linear(feature_size, classes)

    def forward(self, x, lengths):
        x = self.noise_input(x)
        x = self.drop_input(x)

        # sort
        lengths_sorted, sort, unsort = self._sort_by(lengths)
        x_sorted = sort(x)

        # encode
        outputs, last_outputs = self.encoder(x_sorted, lengths_sorted)
        representations, attentions = self.attention(outputs, lengths_sorted)

        # unsort
        representations = unsort(representations)

        logits = self.classifier(representations)

        return logits, attentions
