"""

Model Configurations

"""

# this should be a point of reference
MASTER = {
    "name": "TASK1_VREG",
    "op_mode": "char",  # ["char", "word"]
    "batch_train": 32,
    "batch_eval": 1024,
    "epochs": 100,
    "embeddings_file": "word2vec_300_5",
    "embeddings_dim": 300,
    "embeddings_project": False,
    "embeddings_project_dim": 100,
    "embeddings_trainable": False,
    "embeddings_skip": False,
    "input_noise": 0.2,
    "input_dropout": 0.2,
    "encoder_dropout": 0.2,
    "encoder_size": 256,
    "encoder_layers": 1,
    "encoder_type": "att-rnn",
    "attention_layers": 1,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "rnn_type": "LSTM",
    "rnn_bidirectional": True,
    "base": None,
    "patience": 10,
    "weight_decay": 0.0,
    "clip_norm": 5,
}
