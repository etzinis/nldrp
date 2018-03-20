from hyperopt import hp

HP_EMOTION = {
    "encoder_size": hp.choice('encoder_size', [128, 256]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),
    "input_noise": hp.choice('input_noise', [0.2, 0.5, 0.7]),
    "input_dropout": hp.choice('input_dropout', [0.2, 0.4, 0.8]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.3, 0.5]),
}
