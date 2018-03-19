import os
from copy import deepcopy

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nldrp.dnn.config import DNN_BASE_PATH


def compute_metrics(Y_predicted, Y_true):
    uw_f1 = f1_score(Y_predicted, Y_true, average='macro')
    w_acc = accuracy_score(Y_predicted, Y_true)
    cmat = confusion_matrix(Y_true, Y_predicted)
    with np.errstate(divide='ignore'):
        uw_acc = (cmat.diagonal() / (1.0 * cmat.sum(axis=1) + 1e-6
                                     )).mean()
        if np.isnan(uw_acc):
            uw_acc = 0.
    metrics_l = [('uw_f1', uw_f1),
                 ('uw_acc', uw_acc),
                 ('w_acc', w_acc)]
    metric_dic = dict(metrics_l)
    return metric_dic


def generate_speaker_splits(features_dic):
    sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']

    for session in sessions:
        for val_gender, test_gender in [("M", "F"), ("F", "M")]:
            val_speaker = session + val_gender
            test_speaker = session + test_gender

            train_speakers = [x for x in sorted(features_dic.keys())
                              if x not in [test_speaker, val_speaker]]

            yield train_speakers, val_speaker, test_speaker


def generate_cross_norm_folds(features_dic):
    ind_dic = deepcopy(features_dic)

    ##############################################
    # Fit Normalizer on all speakers
    ##############################################
    xs = []
    for speaker, utterances in ind_dic.items():
        for utt_name, utt_data in utterances.items():
            xs.append(utt_data["x"])
    normalizer = StandardScaler().fit(np.concatenate(xs, axis=0))

    ##############################################
    # Generate splits
    ##############################################
    for split in generate_speaker_splits(ind_dic):
        train_speakers, val_speaker, test_speaker = split

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        X_val = []
        y_val = []

        # training data
        for train_speaker in train_speakers:
            for utt_name, utt_data in ind_dic[train_speaker].items():
                X_train.append(utt_data["x"])
                y_train.append(utt_data["y"])

        # validation data
        for utt_name, utt_data in ind_dic[val_speaker].items():
            X_val.append(utt_data["x"])
            y_val.append(utt_data["y"])

        # testing data
        for utt_name, utt_data in ind_dic[test_speaker].items():
            X_test.append(utt_data["x"])
            y_test.append(utt_data["y"])

        # normalization
        X_train = [normalizer.transform(x) for x in X_train]
        X_val = [normalizer.transform(x) for x in X_val]
        X_test = [normalizer.transform(x) for x in X_test]

        # speaker dependent
        # for each speaker normalize its features independently

        yield X_train, X_val, X_test, y_train, y_val, y_test


def generate_speaker_dependent_folds(features_dic):
    norm_per_sp_dic = deepcopy(features_dic)
    for sp, data in norm_per_sp_dic.items():
        this_scaler = StandardScaler().fit(data['x'])
        norm_per_sp_dic[sp]['x'] = this_scaler.transform(data['x'])

    sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']

    for ses in sessions:
        te_speaker = ses + 'M'
        val_speaker = ses + 'F'
        te_data = norm_per_sp_dic[te_speaker]
        val_data = norm_per_sp_dic[val_speaker]
        x_te_list = [te_data['x'], val_data['x']]
        Y_te = te_data['y'] + val_data['y']
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in norm_per_sp_dic.items():
            if tr_speaker == te_speaker or tr_speaker == val_speaker:
                continue
            x_tr_list.append(tr_data['x'])
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        X_te = np.concatenate(x_te_list, axis=0)
        print "Dependent train: {}".format(X_tr.shape)
        print "Dependent test: {}".format(X_te.shape)

        yield X_te, Y_te, X_tr, Y_tr


def generate_speaker_independent_folds(features_dic):
    ind_dic = deepcopy(features_dic)

    sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']

    for ses in sessions:
        te_speaker = ses + 'M'
        val_speaker = ses + 'F'
        te_data = ind_dic[te_speaker]
        val_data = ind_dic[val_speaker]
        x_te_list = [te_data['x'], val_data['x']]
        Y_te = te_data['y'] + val_data['y']
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in ind_dic.items():
            if tr_speaker == te_speaker or tr_speaker == val_speaker:
                print "Ignoring {}".format(tr_speaker)
                continue
            x_tr_list.append(tr_data['x'])
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        X_te = np.concatenate(x_te_list, axis=0)

        tr_scaler = StandardScaler().fit(X_tr)
        X_tr = tr_scaler.transform(X_tr)
        X_te = tr_scaler.transform(X_te)
        print "Independent train: {}".format(X_tr.shape)
        print "Independent test: {}".format(X_te.shape)
        yield X_te, Y_te, X_tr, Y_tr


def dummy_split():
    """
    Generate *dummy* splits in order to verify the DNN pipeline
    :return:
    :rtype:
    """
    IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data',
                                "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5")

    data_dic = joblib.load(IEMOCAP_PATH)

    X = []
    y = []
    for speaker, utterances in data_dic.items():
        for utterance, sample in utterances.items():
            X.append(sample["x"])
            y.append(sample["y"])

    normalizer = StandardScaler()
    normalizer.fit(np.concatenate(X, axis=0))

    X = [normalizer.transform(x) for x in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test
