import os

from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split


def tune_se20174a():
    DATA_DIR = os.path.join(BASE_PATH, 'data')
    train = load_data_from_dir(os.path.join(DATA_DIR, 'sentiment2017'))
    X = [obs[1] for obs in train]
    y = [obs[0] for obs in train]

    model_config = SEMEVAL_2017

    # pass a transformer function, for preparing tha labels for training
    label_map = {label: idx for idx, label in enumerate(sorted(list(set(y))))}
    inv_label_map = {v: k for k, v in label_map.items()}
    transformer = LabelTransformer(label_map, inv_label_map)

    def train(params):
        model_config.update(params)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            stratify=y)

        trainer = classifier(config=model_config, name=None,
                             disable_cache=True,
                             X_train=X_train, X_test=X_test,
                             y_train=y_train,
                             y_test=y_test,
                             label_transformer=transformer)

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["f1"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


def tune_emotion():
    # load data and config

    def train(params):
        pass

    return train
