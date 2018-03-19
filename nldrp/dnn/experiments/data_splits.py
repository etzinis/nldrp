import os

import numpy
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nldrp.dnn.config import DNN_BASE_PATH

IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data',
                            "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5")


def dummy_split():
    """
    Generate *dummy* splits in order to verify the DNN pipeline
    :return:
    :rtype:
    """
    data_dic = joblib.load(IEMOCAP_PATH)

    X = []
    y = []
    for speaker, utterances in data_dic.items():
        for utterance, sample in utterances.items():
            X.append(sample["x"])
            y.append(sample["y"])

    normalizer = StandardScaler()
    normalizer.fit(numpy.concatenate(X, axis=0))

    X = [normalizer.transform(x) for x in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test
