"""

Model Configurations

"""

EMOTION_CONFIG = {
    "name": "EMOTION",
    "batch_train": 32,
    "batch_eval": 32,
    "epochs": 10,
    "input_noise": 0.2,
    "input_dropout": 0.5,
    "encoder_dropout": 0.5,
    "encoder_size": 150,
    "encoder_layers": 1,
    "encoder_type": "LSTM",
    "attention_layers": 1,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "bidirectional": False,
    "base": None,
    "patience": 10,
    "weight_decay": 0.0,
    "clip_norm": 1,
}
