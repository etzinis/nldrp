import os

from nldrp.dnn.config import DNN_BASE_PATH
from nldrp.dnn.experiments.data_splits import generate_cross_norm_folds
from sklearn.externals import joblib

# Load the datasets ####################################
from nldrp.dnn.experiments.pipeline import get_model_trainer
from nldrp.dnn.experiments.configs import EMOTION

config = EMOTION

IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data',
                            "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5")

features_dic = joblib.load(IEMOCAP_PATH)
for fold in generate_cross_norm_folds(features_dic):
    X_train, X_test, y_train, y_test = fold

    trainer = get_model_trainer(X_train, X_test, y_train, y_test, config)

    print("Training...")
    for epoch in range(config["epochs"]):
        trainer.model_train()
        trainer.model_eval()
        print()

        trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break
