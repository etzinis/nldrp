import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

from util.boiler import load_embeddings, load_datasets, \
    pipeline_classification, load_pretrained_model
from util.multi_gpu import get_new_gpu_id, get_gpu_id
from util.training import Trainer, Checkpoint, EarlyStop, \
    class_weigths


def get_pretrained(pretrained):
    pretrained_models = None
    pretrained_config = None

    if pretrained is not None:
        if isinstance(pretrained, list):
            pretrained_models = []
            for pt in pretrained:
                pretrained_model, pretrained_config = load_pretrained_model(pt)
                pretrained_models.append(pretrained_model)
        else:
            pretrained_model, pretrained_config = load_pretrained_model(
                pretrained)
            pretrained_models = pretrained_model

    return pretrained_models, pretrained_config


def classifier(config, name,
               X_train, X_test, y_train, y_test,
               ordinal=False,
               pretrained=None,
               finetune=None,
               label_transformer=None,
               disable_cache=False):
    pretrained_models, pretrained_config = get_pretrained(pretrained)

    word2idx = None

    if pretrained_config is not None:
        _config = pretrained_config
    else:
        _config = config

    if _config["op_mode"] == "word":
        word2idx, idx2word, embeddings = load_embeddings(_config)

    # construct the pytorch Datasets and Dataloaders
    train_set, val_set = load_datasets(X_train, y_train,
                                       X_test, y_test,
                                       op_mode=_config["op_mode"],
                                       params=None if disable_cache else name,
                                       word2idx=word2idx,
                                       emojis=_config["emojis"],
                                       label_transformer=label_transformer)

    ########################################################################
    # MODEL
    # Define the model that will be trained and its parameters
    ########################################################################
    classes = len(set(y_train))

    num_embeddings = None

    if _config["op_mode"] == "char":
        num_embeddings = len(train_set.char2idx) + 1
        embeddings = None

    model = GenericModel(embeddings=embeddings,
                         out_size=1 if classes == 2 else classes,
                         num_embeddings=num_embeddings,
                         pretrained=pretrained_models,
                         finetune=finetune,
                         **_config)
    print(model)
    weights = class_weigths(train_set.labels, to_pytorch=True)
    if torch.cuda.is_available():
        model.cuda(get_new_gpu_id())
        weights = weights.cuda()

    # depending on the number of classes, we should use a different loss
    if classes > 2:
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 weight_decay=config["weight_decay"])
    pipeline = pipeline_classification(criterion, binary=classes == 2)

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
                      val_set=val_set,
                      config=config,
                      optimizer=optimizer,
                      pipeline=pipeline,
                      metrics=metrics,
                      train_batch_size=config["batch_train"],
                      eval_batch_size=config["batch_eval"],
                      use_exp=True,
                      inspect_weights=False,
                      checkpoint=Checkpoint(name=name,
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


def training_vanilla(trainer, epochs):
    print("Training...")
    for epoch in range(epochs):
        trainer.model_train()
        trainer.model_eval()
        print()

        trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break
