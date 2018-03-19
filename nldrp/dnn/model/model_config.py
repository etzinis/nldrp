class Config(object):
    def to_dict(self):
        own = {k: v for k, v in self.__class__.__dict__.items()
               if k[:2] != "__"}
        base = {k: v for k, v in self.__class__.__bases__[0].__dict__.items()
                if k[:2] != "__"}

        base.update(own)

        return base


class BasicModelCFG(Config):
    """
    This is the default model
    It is advised to inherit this class and override it properties or add new
    """
    name = 'BasicModel'
    batch_train = 32
    batch_eval = 512
    epochs = 200
    embeddings_file = "word2vec_300_5"
    embeddings_dim = 300
    embeddings_project = False
    embeddings_project_dim = 100
    embeddings_trainable = False
    input_noise = 0.2
    input_dropout = 0.2
    encoder_dropout = 0.4
    encoder_size = 256
    encoder_layers = 1
    encoder_type = "att-rnn"
    rnn_type = "LSTM"
    rnn_bidirectional = True
