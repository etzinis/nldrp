import json
import sys
from copy import deepcopy

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import os
import argparse
import pickle

import pandas

from nldrp.dnn.config import DNN_BASE_PATH
from sklearn.externals import joblib

# Load the datasets ####################################
from nldrp.dnn.models.configs import EMOTION_CONFIG
from nldrp.dnn.models.data_splits import generate_folds
from nldrp.dnn.models.pipeline import get_model_trainer

##############################################################################
# Command line Arguments
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--task', help='task', default="independent")
parser.add_argument('--features', help='features', required=True)
args = parser.parse_args()
print(args)

##############################################################################
# Model Tuning
##############################################################################

config = EMOTION_CONFIG

if args.features == "fused":
    features = "IEMOCAP_emobase2010_rqa.dat"
elif args.features == "linear":
    features = "IEMOCAP_linear_emobase2010_segl_1.0_segol_0.5"
else:
    raise ValueError("invalid features argument")

IEMOCAP_PATH = os.path.join(DNN_BASE_PATH, 'data', features)

features_dic = joblib.load(IEMOCAP_PATH)
scores = []
for i, fold in enumerate(generate_folds(features_dic, args.task), 1):
    X_train, X_val, X_test, y_train, y_val, y_test = fold

    X_train += X_val
    y_train += y_val
    trainer = get_model_trainer(X_train, X_test, y_train, y_test, config)

    print("-" * 40)
    print("RUN: ", i)
    print("-" * 40)
    for epoch in range(config["epochs"]):
        trainer.model_train()
        trainer.model_eval()

        # trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break

    _scores = deepcopy(trainer.scores)
    _scores["best"] = trainer.early_stopping.best
    scores.append(_scores)
    print
    print("BEST:", trainer.early_stopping.best)
    print

with open('scores_{}.pickle'.format(args.task), 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("scores_{}.json".format(args.task), 'w') as f:
    json.dump(scores, f)

data = pandas.DataFrame(scores)
with open("scores_{}.csv".format(args.task), 'w') as f:
    data.to_csv(f, sep=',', encoding='utf-8')

print(
    "AVERAGE:{}".format(sum([score["best"] for score in scores]) / len(scores)))
