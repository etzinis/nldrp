import glob
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hyperopt import tpe
from hyperopt.plotting import main_plot_history, main_plot_histogram


def plot_vars(trials):
    for param in trials.idxs.keys():
        params = [trial["result"]["params"][param]
                  for trial in trials.trials if
                  trial["result"]["status"] == "ok"]
        losses = [trial["result"]["loss"] for trial in trials.trials]

        records = []
        for p, l in zip(params, losses):
            records.append({"loss": l, param: p})

        df = pd.DataFrame(records)

        # sns.swarmplot(x=param, y="loss", data=df)
        # plt.show()
        sns.violinplot(x=param, y="loss", data=df)
        plt.show()


def analysis(name):
    files = glob.glob("{}*.hyperopt".format(name))
    trials = None
    for file in files:
        opt_data = pickle.load(open(file, "rb"))
        _trials = opt_data["trials"]

        if trials is None:
            trials = _trials
        else:
            for attr, value in _trials.__dict__.items():
                if isinstance(getattr(_trials, attr),
                              list) and not attr.startswith("__"):
                    setattr(trials, attr,
                            getattr(trials, attr) + getattr(_trials, attr))

        print()
    main_plot_history(trials=trials, algo=tpe.suggest, )
    main_plot_histogram(trials=trials, algo=tpe.suggest, )
    plot_vars(trials)
    # trial_curves()


# analysis("trials/task3a")
