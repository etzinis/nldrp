import pickle

from hyperopt import fmin, tpe, Trials
from hyperopt import rand


def top_trial(trials, top=None):
    if top is None:
        top = len(trials)
    for trial in sorted(trials.trials, key=lambda k: k['result']['loss'])[:top]:
        if trial["result"]["status"] != "fail":
            loss = trial["result"]["loss"]
            params = trial["result"]["params"]
            print("{:.4f}".format(loss), params)


def hyperopt_run_trials(name, model, params, random_trials=10):
    """

    Args:
        name ():
        model ():
        params ():
        max_trials (): initial max_trials
        random_trials ():

    Returns:

    """
    # try to load an already saved trials object, and increase the max
    try:
        opt_data = pickle.load(open(name + '.hyperopt', 'rb'))
        trials = opt_data["trials"]
        print('Found {} saved Trials!'.format(len(trials.trials)))
    except:
        trials = Trials()

    top_trial(trials, top=5)

    if len(trials.trials) < random_trials:
        print("Using RANDOM search.")
        algorithm = rand.suggest
    else:
        print("Using TPE search.")
        algorithm = tpe.suggest

    best = fmin(fn=model,
                space=params,
                algo=algorithm,
                max_evals=len(trials.trials) + 1,
                trials=trials)

    with open(name + ".hyperopt", "wb") as f:
        print("saving trials...", end=" ")
        opt_data = {
            "trials": trials,
            "space": params,
        }
        pickle.dump(opt_data, f)
        print("done!")
