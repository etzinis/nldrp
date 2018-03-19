import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

# Load the datasets ####################################
from nldrp.dnn.modules.dataloading import EmotionDataset
from nldrp.dnn.modules.models import EmotionModel
from nldrp.dnn.util.boiler import pipeline_classification
from nldrp.dnn.util.training import LabelTransformer, Trainer, Checkpoint, \
    EarlyStop, class_weigths


def get_model_trainer(X_train, X_test, y_train, y_test, config):
    ########################################################################
    # MODEL
    # Define the model that will be trained and its parameters
    ########################################################################

    # pass a transformer function, for preparing tha labels for training
    label_map = {label: idx for idx, label in
                 enumerate(sorted(list(set(y_train))))}
    inv_label_map = {v: k for k, v in label_map.items()}
    transformer = LabelTransformer(label_map, inv_label_map)

    train_set = EmotionDataset(X_train, y_train, label_transformer=transformer)
    test_set = EmotionDataset(X_test, y_test, label_transformer=transformer)

    ########################################################################
    # MODEL
    # Define the model that will be trained and its parameters
    ########################################################################
    classes = len(set(y_train))
    input_size = X_train[0][0].size

    model = EmotionModel(input_size, classes, **config)
    print(model)
    weights = class_weigths(y_train, to_pytorch=True)
    if torch.cuda.is_available():
        model.cuda()
        weights = weights.cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 weight_decay=config["weight_decay"])

    metrics = {
        "acc": lambda y, y_hat: accuracy_score(y, y_hat),
        "precision": lambda y, y_hat: precision_score(y, y_hat,
                                                      average='macro'),
        "recall": lambda y, y_hat: recall_score(y, y_hat, average='macro'),
        "f1": lambda y, y_hat: f1_score(y, y_hat, average='macro'),
    }
    monitor = "f1"

    trainer = Trainer(model=model,
                      task="clf",
                      train_set=train_set,
                      val_set=test_set,
                      config=config,
                      optimizer=optimizer,
                      pipeline=pipeline_classification(criterion),
                      metrics=metrics,
                      train_batch_size=config["batch_train"],
                      eval_batch_size=config["batch_eval"],
                      use_exp=True,
                      inspect_weights=False,
                      checkpoint=Checkpoint(name=config["name"],
                                            model=model,
                                            model_conf=config,
                                            keep_best=True,
                                            scorestamp=True,
                                            metric=monitor,
                                            mode="max",
                                            base=config["base"]),
                      early_stopping=EarlyStop(metric=monitor,
                                               mode="max",
                                               patience=config[
                                                   "patience"])
                      )
    return trainer
