"""!
\brief Utterance level classification schema by concatenating vectors
and performing classification

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import tabulate
# import elm
import itertools
import gc
import pandas as pd
import argparse
import numpy as np
import pprint
import json
import os
import copy
import sys

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, nldrp_dir)
import nldrp.feature_extraction.pipeline.utterance_feature_loader as \
    feature_loader
import nldrp.config


def generate_speaker_independent_folds(features_dic):
    all_X = np.concatenate([v['x'] for k, v in features_dic.items()],
                           axis=0)
    all_scaler = StandardScaler().fit(all_X)

    for te_speaker, te_data in features_dic.items():
        x_te = all_scaler.transform(te_data['x'])
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in features_dic.items():
            if tr_speaker == te_speaker:
                continue
            sp_x = all_scaler.transform(tr_data['x'])
            x_tr_list.append(sp_x)
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        yield x_te, te_data['y'], X_tr, Y_tr


def generate_speaker_dependent_folds(features_dic,
                                     n_splits=5,
                                     random_seed=7):
    norm_per_sp_dic = copy.deepcopy(features_dic)
    for sp, data in norm_per_sp_dic.items():
        this_scaler = StandardScaler().fit(data['x'])
        norm_per_sp_dic[sp]['x'] = this_scaler.transform(data['x'])

    xy_l = [v for (sp, v) in norm_per_sp_dic.items()]
    x_all = np.concatenate([v['x'] for v in xy_l])
    y_all = [utt_label for speaker_labels in [v['y'] for v in xy_l]
             for utt_label in speaker_labels]

    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=random_seed)
    for tr_ind, te_ind in skf.split(x_all, y_all):
        yield (x_all[te_ind],
               [y_all[i] for i in te_ind],
               x_all[tr_ind],
               [y_all[i] for i in tr_ind])


def generate_speaker_folds(features_dic):
    for te_speaker, te_data in features_dic.items():
        x_tr_list = []
        Y_tr = []
        for tr_speaker, tr_data in features_dic.items():
            if tr_speaker == te_speaker:
                continue
            x_tr_list.append(tr_data['x'])
            Y_tr += tr_data['y']

        X_tr = np.concatenate(x_tr_list, axis=0)
        yield te_speaker, te_data['x'], te_data['y'], X_tr, Y_tr


def compute_metrics(Y_predicted, Y_true):
    uw_f1 = f1_score(Y_predicted, Y_true, average='macro')
    w_f1 = f1_score(Y_predicted, Y_true, average='micro')

    uw_rec = recall_score(Y_predicted, Y_true, average='macro')
    w_rec = recall_score(Y_predicted, Y_true, average='micro')

    uw_prec = precision_score(Y_predicted, Y_true, average='macro')
    w_prec = precision_score(Y_predicted, Y_true, average='micro')

    w_acc = accuracy_score(Y_predicted, Y_true)
    cmat = confusion_matrix(Y_true, Y_predicted)
    with np.errstate(divide='ignore'):
        uw_acc = (cmat.diagonal() / (1.0 * cmat.sum(axis=1) + 1e-6
                                     )).mean()
        if np.isnan(uw_acc):
            uw_acc = 0.

    metrics_l = [('uw_f1', uw_f1),
                 #('w_f1', w_f1),
                 # ('uw_rec', uw_rec),
                 # ('w_rec', w_rec),
                 # ('uw_prec', uw_prec),
                 # ('w_prec', w_prec),
                 ('uw_acc', uw_acc),
                 ('w_acc', w_acc)]

    metric_dic = dict(metrics_l)
    return metric_dic


def configure_models():
    # class ELMWrapper(object):
    #     def __init__(self, **kwargs):
    #         self.kernel = elm.ELMKernel()
    #     def predict(self, x):
    #         return self.kernel.test(x)
    #     def fit(self, x_tr, y_tr):
    #         self.le = LabelEncoder()
    #         self.le.fit(y_tr)
    #         int_labels = self.le.transform(y_tr)
    #         labels_col = np.asarray(int_labels)
    #         labels_col = np.reshape(labels_col, (-1,1))
    #         new_data = np.concatenate([labels_col, x_tr], axis=1)
    #
    #         new_data = elm.read('/home/thymios/Desktop/iris.data')
    #         print new_data.shape
    #
    #         self.kernel.search_param(new_data,
    #                                  of="accuracy",
    #                                  eval=10)
    #         # self.kernel.train(new_data)
    #         exit()

    models = []
    # models.append(('ELM', ELMWrapper()))
    models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # models.append(('RF', RandomForestClassifier()))
    # models.append(('ADAb', AdaBoostClassifier()))
    # models.append(('GRADb', GradientBoostingClassifier()))
    # models.append(('QDA', QuadraticDiscriminantAnalysis()))
    # models.append(('LinR', LogisticRegression()))


    return dict(models)


def dummy_generate_SVMs_and_LRs():
    svm_params = [('svm', c) for c in [0.1, 0.3, 0.5, 1, 3, 5, 7, 8, 10]]
    lr_params = [('lr', c) for c in [1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 3]]
    all_params = svm_params + lr_params

    for m_name, c in all_params:
        desc = '{}_{}'.format(m_name, str(c))
        if m_name == 'svm':
            yield desc, SVC(C=c)
        else:
            yield desc, LogisticRegression(C=c)


def speaker_dependent(model,
                      X_te, Y_te,
                      X_tr, Y_tr):
    n_components = int(X_tr.shape[1] / 2)
    # n_components = 3000
    # pca = PCA(n_components=n_components).fit(X_tr)
    #
    # X_tr = pca.transform(X_tr)
    # FIXME: Per speaker normalization the dirty way

    model.fit(X_tr, Y_tr)
    # X_te = pca.transform(X_te)

    Y_pred = model.predict(X_te)
    model_metrics = compute_metrics(Y_pred, Y_te)
    return model_metrics


def speaker_independent(model,
                      X_te, Y_te,
                      X_tr, Y_tr):
    n_components = int(X_tr.shape[1] / 10)
    n_components = 3000
    # pca = PCA(n_components=n_components).fit(X_tr)
    # X_tr = pca.transform(X_tr)
    # X_te = pca.transform(X_te)

    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_te)
    model_metrics = compute_metrics(Y_pred, Y_te)
    return model_metrics


def evaluate_loso(features_dic):
    all_models = list(dummy_generate_SVMs_and_LRs()) #configure_models()
    result_dic = {}
    all_results = {}
    folds_independent = list(generate_speaker_independent_folds(features_dic))
    folds_dependent = list(generate_speaker_dependent_folds(features_dic))

    #for model_name, model in all_models.items():
    for model_name, model in all_models:
        result_dic[model_name] = {}

        for X_te, Y_te, X_tr, Y_tr in folds_dependent:
            exp = 'dependent'
            m = {}
            m[exp] = speaker_dependent(
                model, X_te,
                Y_te, X_tr, Y_tr)

            for k, v in m[exp].items():
                col_name = exp + '_' + k
                if result_dic[model_name] and col_name in result_dic[model_name]:
                     result_dic[model_name][col_name].append(v)
                else:
                    result_dic[model_name][col_name]=[v]


        for X_te, Y_te, X_tr, Y_tr in folds_independent:
            exp = 'independent'
            m = {}
            m[exp] = speaker_independent(
                model, X_te,
                Y_te, X_tr, Y_tr)

            for k, v in m[exp].items():
                col_name = exp + '_' + k
                if result_dic[model_name] and col_name in result_dic[model_name]:
                     result_dic[model_name][col_name].append(v)
                else:
                    result_dic[model_name][col_name]=[v]

        print model_name
        for k, v in result_dic[model_name].items():
            result_dic[model_name][k] = round(np.mean(v), 4)
        pprint.pprint(result_dic[model_name])

    all_results['model'] = []
    for k in result_dic[model_name]:
        for mod, _ in all_models:
            if mod not in all_results['model']:
                all_results['model'].append(mod)
            if mod in result_dic:
                if k in all_results:
                    all_results[k].append(result_dic[mod][k])
                else:
                    all_results[k] = [result_dic[mod][k]]

    #df = pd.DataFrame.from_dict(all_results)
    #df.to_clipboard()
    return all_results


def convert_2_numpy_per_utterance(dataset_dic):
    converted_dic = {}
    for spkr in dataset_dic:
        x_list = []
        y_list = []
        converted_dic[spkr] = {}
        for id, el_dic in dataset_dic[spkr].items():
            label = el_dic['y']
            feat_vec = el_dic['x']
            x_list.append(feat_vec)
            y_list.append(label)

        this_utt_array = np.array(x_list)
        converted_dic[spkr]['x']=this_utt_array
        converted_dic[spkr]['y']=y_list

    return converted_dic

def nl_feature_load(list_of_paths):
    nl_features = {}
    for nl_feat_p in list_of_paths:
        the_path = os.path.join(nldrp.config.NL_FEATURE_PATH, nl_feat_p)
        temp_dic = joblib.load(the_path)
        nl_features[nl_feat_p] = temp_dic
    print "Read {} features from {}".format(len(nl_features.items()), len(list_of_paths))
    return nl_features


def fusion_loso(list_of_paths):
    all_results = {}
    emo_data_dic = joblib.load(nldrp.config.EMOBASE_PATH)
    nl_feature_dic = nl_feature_load(list_of_paths)
    for nl_feat_p, temp_dic in nl_feature_dic.items():
        print nl_feat_p
        final_data_dic = copy.deepcopy(emo_data_dic)
        print "COPY"
        try:
            for spkr in temp_dic:
                for id, el_dic in temp_dic[spkr].items():
                    assert el_dic['y'] == final_data_dic[spkr][id]['y']
                    prev_vec = final_data_dic[spkr][id]['x']
                    this_vec = el_dic['x']
                    new_vec = np.concatenate([prev_vec, this_vec], axis=0)
                    final_data_dic[spkr][id]['x'] = new_vec
        except Exception as e:
            print "Failed to update the Fused dictionary"
            raise e

        fused_converted_dic = convert_2_numpy_per_utterance(final_data_dic)

        print "FUSE"

        results = evaluate_loso(fused_converted_dic)
        print "EVALUATE"

        all_results[nl_feat_p] = results
        formatted_results = {'configs': []}
        for k, v in all_results.items():
            for item, lst in v.items():
                cnt = len(lst)
                if item in formatted_results:
                    formatted_results[item] += lst
                else:
                    formatted_results[item] = lst
            for _ in range(cnt):
                formatted_results['configs'].append(k)

        print "AGGREGATE RESULTS"

        with open(os.path.join(nldrp.config.BASE_PATH, 'up2now_best_features_kl.json'), 'w') as fd:
            json.dump(formatted_results, fd)

        print "JSON DUMP"

        gc.collect()
        #df = pd.DataFrame.from_dict(formatted_results)
        #df.to_csv(os.path.join(nldrp.config.BASE_PATH, 'up2now_best_features.csv'))

    print "FINISHED"

    formatted_results = {'configs': []}
    for k, v in all_results.items():
        for item, lst in v.items():
            cnt = len(lst)
            if item in formatted_results:
                formatted_results[item].append(lst)
            else:
                formatted_results[item] = [lst]
        for _ in range(cnt):
            formatted_results['configs'].append(k)

    print "FORMATTED RESULTS"
    try:
        with open(os.path.join(nldrp.config.BASE_PATH, 'best_features_kl.json'), 'w') as fd:
            json.dump(formatted_results, fd)
        df = pd.DataFrame.from_dict(formatted_results)
        df.to_csv(os.path.join(nldrp.config.BASE_PATH, 'best_features_kl.csv'))
    except Exception as e:
         with open(os.path.join(nldrp.config.BASE_PATH, 'best_features_kl.json'), 'w') as fd:
            json.dump(formatted_results, fd)

    return all_results


def get_args():
    """! Command line parser for Utterance level classification Leave
    one speaker out schema pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level classification Leave one '
                    'speaker out schema pipeline' )
    parser.add_argument('-i', '--input_features_paths', nargs='+',
                        help='File paths of the features you want to '
                             'concatenate and the classify')

    # parser.add_argument("--dataset", type=str,
    #                     help="""The name of the dataset""",
    #                     required=True,
    #                     choices=['SAVEE'])
    # parser.add_argument("-i", "--save_dir", type=str,
    #     help="""Where the corresponding binary file full of
    #     data that will contain the dictionary for each speaker is
    #     stored.
    #     Another subdic for all the sentences with their ids
    #     and a 1d numpy matrix for each one of them.
    #     """,
    #     default=nldrp.config.EXTRACTED_FEATURES_PATH )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    fusion_loso(os.listdir(nldrp.config.NL_FEATURE_PATH))
