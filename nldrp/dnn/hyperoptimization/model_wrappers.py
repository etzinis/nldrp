import os

from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split

from config import BASE_PATH
from data.rest import load_isear
from data.task1 import parse as parse1
from data.task2 import load_task2
from data.task3 import parse as parse3a
from model.configs import TASK3_A, SEMEVAL_2017, ISEAR, TASK1_EC, \
    TASK1_ELREG, TASK1_VREG, TASK3_B, TASK1_ELOC, TASK1_VOC, TASK2_A
from model.model_pipelines import classifier, multi_classifier, \
    regressor
from model.nbow_baseline.load_data import load_data_from_dir
from util.training import LabelTransformer


def tune_task_3a():
    X, y = parse3a(task="a")
    model_config = TASK3_A

    def train(params):
        model_config.update(params)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            stratify=y)

        trainer = classifier(config=model_config, name=None,
                             disable_cache=True,
                             X_train=X_train, X_test=X_test,
                             y_train=y_train,
                             y_test=y_test, )

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


def tune_task_3b():
    X, y = parse3a(task="b")
    model_config = TASK3_B

    def train(params):
        model_config.update(params)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            stratify=y)
        name = model_config["name"] + "_" + model_config["op_mode"]
        trainer = classifier(config=model_config, name=name,
                             disable_cache=True,
                             X_train=X_train, X_test=X_test,
                             y_train=y_train,
                             y_test=y_test, )

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()
                trainer.checkpoint.check()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["f1"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


def tune_task_2a():
    X_train, y_train = load_task2("train")
    X_test, y_test = load_task2("trial")
    model_config = TASK2_A

    def train(params):
        model_config.update(params)

        name = "_".join([model_config["name"], model_config["encoder_type"],
                         model_config["op_mode"]])
        trainer = classifier(config=model_config, name=name,
                             disable_cache=False,
                             X_train=X_train, X_test=X_test,
                             y_train=y_train,
                             y_test=y_test, )

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()
                trainer.checkpoint.check()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["f1"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


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


def tune_isear():
    X_train, X_test, y_train, y_test = load_isear()

    model_config = ISEAR

    # pass a transformer function, for preparing tha labels for training
    label_map = {label: idx for idx, label in
                 enumerate(sorted(list(set(y_train))))}
    inv_label_map = {v: k for k, v in label_map.items()}
    transformer = LabelTransformer(label_map, inv_label_map)

    def train(params):
        model_config.update(params)

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


def tune_task_1_e_c():
    X_train, y_train, X_test, y_test = parse1(task='E-c')
    model_config = TASK1_EC

    def train(params):
        model_config.update(params)
        name = model_config["name"] + "_" + model_config["op_mode"]
        trainer = multi_classifier(config=model_config,
                                   name=name,
                                   X_train=X_train, X_test=X_test,
                                   y_train=y_train,
                                   y_test=y_test, )

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()
                trainer.checkpoint.check()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["jaccard"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


def tune_task_1_ei_reg(emotion):
    X_train, y_train, X_test, y_test = parse1(task='EI-reg',
                                              emotion=emotion)
    y_train = [y[1] for y in y_train]
    y_test = [y[1] for y in y_test]
    model_config = TASK1_ELREG

    def train(params):
        model_config.update(params)

        name = "_".join(
            [model_config["name"], emotion, model_config["op_mode"]])
        trainer = regressor(config=model_config,
                            name=name,
                            X_train=X_train, X_test=X_test,
                            y_train=y_train,
                            y_test=y_test, )

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["pearson"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


def tune_task_1_ei_oc(emotion):
    X_train, y_train, X_test, y_test = parse1(task='EI-oc', emotion=emotion)

    y_train = [y[1] for y in y_train]
    y_test = [y[1] for y in y_test]
    model_config = TASK1_ELOC

    def train(params):
        model_config.update(params)

        name = "_".join(
            [model_config["name"], emotion, model_config["op_mode"]])
        trainer = classifier(config=model_config,
                             name=name,
                             ordinal=True,
                             X_train=X_train, X_test=X_test,
                             y_train=y_train,
                             y_test=y_test, )

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["pearson"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


def tune_task_1_v_oc():
    model_config = TASK1_VOC

    X_train, y_train, X_test, y_test = parse1(task='V-oc')

    # keep only scores
    y_train = [str(y[1]) for y in y_train]
    y_test = [str(y[1]) for y in y_test]

    # pass explicit mapping
    label_map = {"-3": 0, "-2": 1, "-1": 2, "0": 3, "1": 4, "2": 5, "3": 6, }
    inv_label_map = {v: int(k) for k, v in label_map.items()}
    transformer = LabelTransformer(label_map, inv_label_map)

    def train(params):
        model_config.update(params)

        name = "_".join(
            [model_config["name"], model_config["op_mode"]])
        trainer = classifier(config=model_config,
                             name=name,
                             ordinal=True,
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
                trainer.checkpoint.check()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["pearson"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


def tune_task_1_v_reg():
    X_train, y_train, X_test, y_test = parse1(task='V-reg')
    y_train = [y[1] for y in y_train]
    y_test = [y[1] for y in y_test]
    model_config = TASK1_VREG

    def train(params):
        model_config.update(params)

        name = "_".join([model_config["name"], model_config["op_mode"]])
        trainer = regressor(config=model_config,
                            name=name,
                            X_train=X_train, X_test=X_test,
                            y_train=y_train,
                            y_test=y_test, )

        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()
                trainer.checkpoint.check()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["pearson"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train
