import os

from nldrp.dnn.config import DNN_BASE_PATH
from nldrp.dnn.experiments.data_splits import generate_speaker_splits

from sklearn.externals import joblib

IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data',
                            "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5")

features_dic = joblib.load(IEMOCAP_PATH)
for split in generate_speaker_splits(features_dic):
    train_speakers, val_speaker, test_speaker = split
    print("-" * 40)
    print("train_speakers: ", train_speakers)
    print("val_speaker: ", val_speaker)
    print("test_speaker: ", test_speaker)
