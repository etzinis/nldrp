import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import argparse
import json

import os

from nldrp.dnn.config import DNN_BASE_PATH
from nldrp.dnn.hyperoptimization.hypertuning import hyperopt_run_trials
from nldrp.dnn.models.configs import EMOTION_CONFIG
from hyperopt import hp, STATUS_OK, STATUS_FAIL
from sklearn.externals import joblib

from nldrp.dnn.models.data_splits import get_split
from nldrp.dnn.models.pipeline import get_model_trainer

##############################################################################
# Command line Arguments
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='run name', required=True)
parser.add_argument('--index', help='for parallel computation', default=None)
parser.add_argument('--conf', help='task', default="independent")
parser.add_argument('--val_speaker', help='validation speaker',
                    default="Ses02F")
args = parser.parse_args()
print(args)

##############################################################################
# Model Tuning
##############################################################################
PARAM_SPACE = {
    "encoder_size": hp.choice('encoder_size', [128, 256]),
    "encoder_layers": hp.choice('encoder_layers', [1, 2]),
    "input_noise": hp.choice('input_noise', [0.2, 0.5, 0.7, 0.9]),
    "input_dropout": hp.choice('input_dropout', [0.3, 0.5, 0.8]),
    "encoder_dropout": hp.choice('encoder_dropout', [0.3, 0.5, 0.8]),
    # "bidirectional": hp.choice('bidirectional', [True, False]),
    # "lr": hp.loguniform('lr', 1e-2, 1e-4),
}


def get_split_by_speaker(splits, speaker):
    return [s for s in splits if s[2] == speaker][0]


def tuning_pipeline():
    model_config = EMOTION_CONFIG

    IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data',
                                "IEMOCAP_emobase2010_rqa.dat")

    ind_dic = joblib.load(IEMOCAP_PATH)
    splits_file = os.path.join(DNN_BASE_PATH, 'hyperoptimization',
                               "dataset_splits.json")

    splits = []
    with open(splits_file, 'r') as f:
        splits = json.load(f, encoding="utf-8")

    train_speakers, val_speaker, test_speaker = get_split_by_speaker(splits,
                                                                     args.val_speaker)

    X_train, X_val, X_test, y_train, y_val, y_test = get_split(train_speakers,
                                                               val_speaker,
                                                               test_speaker,
                                                               ind_dic,
                                                               args.conf)
    X_train += X_val
    y_train += y_val
    
    def train(params):
        print()
        print("-" * 40)
        print(params)
        print("-" * 40)
        model_config.update(params)
        trainer = get_model_trainer(X_train, X_test, y_train, y_test,
                                    model_config)
        try:
            print("Training...")
            for epoch in range(model_config["epochs"]):
                trainer.model_train()
                trainer.model_eval()
                print()

                if trainer.early_stopping.stop():
                    print("Early stopping...")
                    break

            score = max(trainer.scores["un_acc"])
            return {'loss': -score, 'status': STATUS_OK, 'params': params}

        except Exception as e:
            print("There was an error!!!", e)
            return {'loss': 0, 'status': STATUS_FAIL}

    return train


##############################################################################
# Run trials
##############################################################################

name = args.name
name += "_" + str(args.val_speaker)

if args.index is not None:
    name += "_" + str(args.index)

hyperopt_run_trials(name, tuning_pipeline(), PARAM_SPACE, random_trials=10)
