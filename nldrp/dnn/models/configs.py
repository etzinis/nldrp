"""

Model Configurations

"""

EMOTION_CONFIG = {
    "name": "EMOTION",
    "batch_train": 100,
    "batch_eval": 100,
    "epochs": 100,
    "input_noise": 0.0,
    "input_dropout": 0.0,
    "encoder_dropout": 0.5,
    "encoder_size": 256,
    "encoder_layers": 1,
    "encoder_type": "LSTM",
    "attention_layers": 1,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "bidirectional": True,
    "base": None,
    "patience": 10,
    "weight_decay": 0.0,
    "clip_norm": 1,
}
