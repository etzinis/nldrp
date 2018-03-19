"""

Model Configurations

"""

EMOTION = {
    "name": "EMOTION",
    "batch_train": 256,
    "batch_eval": 256,
    "epochs": 100,
    "input_noise": 0.2,
    "input_dropout": 0.0,
    "encoder_dropout": 0.5,
    "encoder_size": 200,
    "encoder_layers": 1,
    "encoder_type": "GRU",
    "attention_layers": 1,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "bidirectional": False,
    "base": None,
    "patience": 10,
    "weight_decay": 0.001,
    "clip_norm": 1,
}
