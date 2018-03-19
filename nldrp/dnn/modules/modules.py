from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.regularization import GaussianNoise


class RecurrentEncoder(nn.Module):
    def __init__(self, input_size,
                 rnn_size,
                 rnn_type,
                 num_layers,
                 bidirectional,
                 dropout):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(RecurrentEncoder, self).__init__()

        if rnn_type == "GRU":
            rnn = nn.GRU
        elif rnn_type == "LSTM":
            rnn = nn.LSTM

        rnns = []
        for i in range(num_layers):

            if i == 0:
                _input_size = input_size
            else:
                _input_size = rnn_size
                if bidirectional:
                    _input_size *= 2

            rnns.append(rnn(input_size=_input_size,
                            hidden_size=rnn_size,
                            num_layers=1,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True))

        self.encoders = nn.ModuleList(rnns)

        # the dropout "layer" for the output of the RNN
        self.drop_rnn = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size

        if bidirectional:
            self.feature_size *= 2

    @staticmethod
    def last_timestep(unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, embs, lengths):

        outputs = []
        for encoder in self.encoders:
            if len(outputs) > 0:
                packed = pack_padded_sequence(outputs[-1],
                                              list(lengths.data),
                                              batch_first=True)
            else:
                packed = pack_padded_sequence(embs,
                                              list(lengths.data),
                                              batch_first=True)
            out_packed, _ = encoder(packed)
            outputs_unpacked, _ = pad_packed_sequence(out_packed,
                                                      batch_first=True)
            outputs_unpacked = self.drop_rnn(outputs_unpacked)
            outputs.append(outputs_unpacked)

        # get the outputs from the last *non-masked* timestep for each sentence
        last_outputs = self.last_timestep(outputs[-1], lengths)

        return outputs[-1], last_outputs


class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 embeddings=None,
                 noise=.0,
                 dropout=.0,
                 trainable=False):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            noise (float):
            dropout (float):
            trainable (bool):
        """
        super(Embed, self).__init__()

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      max_norm=1,
                                      sparse=False)

        # initialize the weights of the Embedding layer,
        # with the given pre-trained word vectors
        if embeddings is not None:
            print("Initializing Embedding layer with pre-trained weights!")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, x):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)

        Returns: the logits for each class

        """
        embeddings = self.embedding(x)

        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings
