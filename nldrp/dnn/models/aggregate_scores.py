import csv
import json
from glob import glob
from pprint import pprint

import pandas
from numpy import mean

files = glob('*.json')
results = {}
for file in files:
    name = file.split(".")[0].split("_")
    name = name[1] + " " + name[2]
    data = json.load(open(file))
    accuracy = mean([max(run["acc"]) for run in data])
    uw_accuracy = mean([max(run["un_acc"]) for run in data])
    f1 = mean([max(run["f1"]) for run in data])
    results[name] = {
        "accuracy": accuracy,
        "uw_accuracy": uw_accuracy,
        "f1": f1,
    }

data = pandas.DataFrame().from_dict(results, orient='index')
with open('results.csv', 'w') as f:
    data.to_csv(f, sep=',', encoding='utf-8')

pprint(results)
