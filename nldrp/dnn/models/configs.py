"""

Model Configurations

"""

EMOTION_CONFIG = {
    "name": "EMOTION",
    "batch_train": 128,
    "batch_eval": 128,
    "epochs": 50,
    "input_noise": 0.5,
    "input_dropout": 0.5,
    "encoder_dropout": 0.5,
    "encoder_size": 256,
    "encoder_layers": 2,
    "encoder_type": "LSTM",
    "attention_layers": 1,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "bidirectional": True,
    "base": None,
    "patience": 10,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "clip_norm": 5,
}
