import glob
import json
import pickle

for file in glob.glob("SEMEVAL*.conf"):
    fname = file.strip(".conf")
    model_conf = pickle.load(open(file, 'rb'))

    with open('{}.json'.format(fname), 'w') as fp:
        json.dump(model_conf, fp, indent=4)
