import argparse

from nldrp.dnn.hyperoptimization.hparams_spaces import HP_EMOTION
from nldrp.dnn.hyperoptimization.hypertuning import hyperopt_run_trials
from nldrp.dnn.hyperoptimization.model_wrappers import tune_emotion

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='run name.', required=True)
parser.add_argument('--emotion', help='for emotion models', default=None)
parser.add_argument('--index', help='for parallel computation', default=None)
args = parser.parse_args()
print(args)

models = {
    "emotion": (tune_emotion, HP_EMOTION),
}

model_wrapper, params_space = models[args.name]
if args.emotion is not None:
    model_train = model_wrapper(args.emotion)
else:
    model_train = model_wrapper()

name = args.name

if args.index is not None:
    name += "_" + str(args.index)

hyperopt_run_trials(name, model_train, params_space, random_trials=10)
