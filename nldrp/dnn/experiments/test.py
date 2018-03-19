import os

from sklearn.externals import joblib

from config import BASE_PATH

file = os.path.join(BASE_PATH, 'data',
                    "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5")

data_dic = joblib.load(file)

X = []
y = []
for speaker, utterances in data_dic.items():
    for utterance, sample in utterances.items():
        X.append(sample["x"])
        y.append(sample["y"])
