from hyperopt import hp

HP_TASK_3A = {
    "embeddings_dim": hp.choice('embeddings_dim', [25, 50]),
    "encoder_size": hp.choice('encoder_size', [128, 256]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),
}

HP_TASK_3B = {
    "embeddings_file": hp.choice('embeddings_file',
                                 ["word2vec_300_6_concatened",
                                  "word2vec_500_6_20_neg"]),
    "rnn_skip": hp.choice('rnn_skip', [True, False]),
    "attention_layers": hp.choice('attention_layers', [1, 2]),
    "attention_activation": hp.choice('attention_activation', ["relu", "tanh"]),
    "attention_context": hp.choice('attention_context', ["none", "mean"]),
    "encoder_size": hp.choice('encoder_size', [150, 250]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.2, 0.4]),
    "input_noise": hp.choice('input_noise', [0.05, 0.07]),
}

HP_TASK_2A = {
    "rnn_skip": hp.choice('rnn_skip', [True, False]),
    "encoder_size": hp.choice('encoder_size', [250, 400]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.3, 0.5]),
    "input_noise": hp.choice('input_noise', [0.05, 0.07]),
    "attention_layers": hp.choice('attention_layers', [1, 2, 3]),
    "attention_dropout": hp.choice('encoder_dropout', [0., 0.2]),
    "attention_activation": hp.choice('encoder_dropout', ["tanh", "relu"]),
    "embeddings_file": hp.choice('embeddings_file',
                                 ["word2vec_300_6_20_neg.txt",
                                  "word2vec_500_6_20_neg.txt",
                                  "400_5_20_twitter.txt"]),
}

HP_ISEAR = {

}

HP_TASK1_EC = {
    "embeddings_skip": hp.choice('embeddings_skip', [True, False]),
    "rnn_skip": hp.choice('rnn_skip', [True, False]),
    "attention_layers": hp.choice('attention_layers', [1, 2]),
    "attention_context": hp.choice('attention_context', ["none", "mean"]),
    "encoder_size": hp.choice('encoder_size', [150, 250]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.2, 0.4]),
}

HP_TASK1_EI_REG = {
    "attention_layers": hp.choice('attention_layers', [1, 2]),
    "attention_context": hp.choice('attention_context', ["none", "mean"]),
    "encoder_size": hp.choice('encoder_size', [150, 250]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.2, 0.4]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),

}

HP_TASK1_EI_OC = {
    "attention_layers": hp.choice('attention_layers', [1, 2]),
    "attention_context": hp.choice('attention_context', ["none", "mean"]),
    "encoder_size": hp.choice('encoder_size', [150, 250]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.2, 0.4]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),
}

HP_TASK1_V_REG = {
    "attention_layers": hp.choice('attention_layers', [1, 2]),
    "attention_context": hp.choice('attention_context', ["none", "mean"]),
    "encoder_size": hp.choice('encoder_size', [150, 250]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.2, 0.4]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),
}

HP_TASK1_V_OC = {
    "embeddings_file": hp.choice('embeddings_file',
                                 ["word2vec_300_6_concatened",
                                  "word2vec_300_6_concatened",
                                  "word2vec_500_6_20_neg"]),
    "rnn_skip": hp.choice('rnn_skip', [True, False]),
    "attention_layers": hp.choice('attention_layers', [1, 2]),
    "attention_context": hp.choice('attention_context', ["none", "mean"]),
    "encoder_size": hp.choice('encoder_size', [150, 250]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.2, 0.4]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),
}
