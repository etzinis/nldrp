import json

import numpy

from util.boiler import load_pretrained_model


def dump_attentions(X, y, dataset, model_name, predictor):
    model, conf = load_pretrained_model(model_name)
    pred, posteriors, attentions, tokenized = predictor(model, conf, X)

    data = []
    for tweet, label, prediction, posterior, attention in zip(tokenized, y,
                                                              pred, posteriors,
                                                              attentions):
        label = numpy.array(label)
        prediction = numpy.array(prediction).astype(label.dtype)

        item = {
            "text": tweet,
            "label": label.tolist(),
            "prediction": prediction.tolist(),
            "posterior": posterior.tolist(),
            "attention": attention.tolist(),
        }
        data.append(item)
    with open("attentions/{}_{}.json".format(model_name, dataset, ), 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))
