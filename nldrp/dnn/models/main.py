import os

from nldrp.dnn.config import DNN_BASE_PATH
from sklearn.externals import joblib

# Load the datasets ####################################
from nldrp.dnn.models.configs import EMOTION
from nldrp.dnn.models.data_splits import generate_folds
from nldrp.dnn.models.pipeline import get_model_trainer

config = EMOTION

IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data',
                            "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5")

features_dic = joblib.load(IEMOCAP_PATH)
for fold in generate_folds(features_dic, "cross"):
    X_train, X_val, X_test, y_train, y_val, y_test = fold

    trainer = get_model_trainer(X_train, X_test, y_train, y_test, config)

    print("Training...")
    for epoch in range(config["epochs"]):
        trainer.model_train()
        trainer.model_eval()
        print

        # trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break
