from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_speaker_splits(features_dic):
    sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']

    for session in sessions:
        for val_gender, test_gender in [("M", "F"), ("F", "M")]:
            val_speaker = session + val_gender
            test_speaker = session + test_gender

            train_speakers = [x for x in sorted(features_dic.keys())
                              if x not in [test_speaker, val_speaker]]

            # print("-" * 40)
            # print("train_speakers", train_speakers)
            # print("val_speaker", val_speaker)
            # print("test_speaker", test_speaker)
            # print("-" * 40)

            yield train_speakers, val_speaker, test_speaker


def map_labels(labels):
    _labels = []
    for label in labels:
        if label == "excited":
            _labels.append("happy")
        else:
            _labels.append(label)
    return _labels


def get_split(train_speakers, val_speaker, test_speaker, ind_dic, conf):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []

    # training data
    for train_speaker in train_speakers:
        speaker_data = []
        for utt_name, utt_data in ind_dic[train_speaker].items():
            speaker_data.append(utt_data["x"])
            y_train.append(utt_data["y"])
        X_train.append(speaker_data)

    # validation data
    for utt_name, utt_data in ind_dic[val_speaker].items():
        X_val.append(utt_data["x"])
        y_val.append(utt_data["y"])

    # testing data
    for utt_name, utt_data in ind_dic[test_speaker].items():
        X_test.append(utt_data["x"])
        y_test.append(utt_data["y"])

    # Normalize across datasets
    if conf == "cross":
        X_train = [utterance for speaker in X_train
                   for utterance in speaker]

        X_all = X_train + X_val + X_test
        normalizer = StandardScaler().fit(np.concatenate(X_all, axis=0))

        X_train = [normalizer.transform(x) for x in X_train]
        X_val = [normalizer.transform(x) for x in X_val]
        X_test = [normalizer.transform(x) for x in X_test]

    # Normalize based on training data
    elif conf == "independent":
        X_train = [utterance for speaker in X_train
                   for utterance in speaker]

        normalizer = StandardScaler().fit(np.concatenate(X_train, axis=0))

        X_train = [normalizer.transform(x) for x in X_train]
        X_val = [normalizer.transform(x) for x in X_val]
        X_test = [normalizer.transform(x) for x in X_test]

    # Normalize per speaker
    elif conf == "dependent":

        for i, speaker in enumerate(X_train):
            xs = np.concatenate(speaker, axis=0)
            normalizer = StandardScaler().fit(xs)
            X_train[i] = [normalizer.transform(x) for x in speaker]

        # flat training data
        X_train = [utterance for speaker in X_train
                   for utterance in speaker]

        # normalizer validation speaker
        val_norm = StandardScaler().fit(np.concatenate(X_val, axis=0))
        X_val = [val_norm.transform(x) for x in X_val]

        # normalizer test speaker
        test_norm = StandardScaler().fit(np.concatenate(X_test, axis=0))
        X_test = [test_norm.transform(x) for x in X_test]

    else:
        raise ValueError("Invalid configuration!")

    y_train = map_labels(y_train)
    y_val = map_labels(y_val)
    y_test = map_labels(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_folds(features_dic, conf):
    """

    Args:
        features_dic: dataset
        conf: valid confs are ["cross", "independent", "dependent"]

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test

    """
    ind_dic = deepcopy(features_dic)

    for split in generate_speaker_splits(ind_dic):
        train_speakers, val_speaker, test_speaker = split
        yield get_split(train_speakers, val_speaker, test_speaker, ind_dic,
                        conf)
