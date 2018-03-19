import argparse

from hyperoptimization.hparams_spaces import (HP_TASK_3A,
                                                          HP_ISEAR, HP_TASK1_EC,
                                                          HP_TASK1_EI_REG,
                                                          HP_TASK1_V_REG,
                                                          HP_TASK_3B,
                                                          HP_TASK1_EI_OC,
                                                          HP_TASK1_V_OC,
                                                          HP_TASK_2A)
from hyperoptimization.model_wrappers import (tune_task_3a,
                                                          tune_isear,
                                                          tune_task_1_e_c,
                                                          tune_task_1_ei_reg,
                                                          tune_task_1_v_reg,
                                                          tune_task_3b,
                                                          tune_task_1_ei_oc,
                                                          tune_task_1_v_oc,
                                                          tune_task_2a)
from util.hypertuning import hyperopt_run_trials

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='run name.', required=True)
parser.add_argument('--emotion', help='for emotion models', default=None)
parser.add_argument('--index', help='for parallel computation', default=None)
args = parser.parse_args()
print(args)

models = {
    "task_3a": (tune_task_3a, HP_TASK_3A),
    "task_2a": (tune_task_2a, HP_TASK_2A),
    "task_3b": (tune_task_3b, HP_TASK_3B),
    "isear": (tune_isear, HP_ISEAR),
    "task_1_e_c": (tune_task_1_e_c, HP_TASK1_EC),
    "task_1_ei_oc": (tune_task_1_ei_oc, HP_TASK1_EI_OC),
    "task_1_ei_reg": (tune_task_1_ei_reg, HP_TASK1_EI_REG),
    "task_1_v_reg": (tune_task_1_v_reg, HP_TASK1_V_REG),
    "task_1_v_oc": (tune_task_1_v_oc, HP_TASK1_V_OC),
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
