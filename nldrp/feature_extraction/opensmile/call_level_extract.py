"""!
\brief Opensmile Call-level extraction by using a specified
configuration file

\warning you are assume to have opensmile already installed and
exported the appropriate paths to your corresponding .bashrc | .zshrc
etc.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

from scipy.io import arff
import argparse
import os
import sys
import numpy as np
from sklearn.externals import joblib
from progress.bar import ChargingBar
import subprocess
import time

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)
import nldrp.config
import nldrp.io.dataloader as dl_savee
import nldrp.io.dataloader_emodb as dl_berlin
import nldrp.io.dataloader_iemo as dl_iemocap


def load_data(dataset_name):
    if dataset_name == 'SAVEE':
        loader_obj = dl_savee.SaveeDataloader(
                      savee_path=nldrp.config.SAVEE_PATH)
        dataset_dic = loader_obj.data_dict
    elif dataset_name == 'IEMOCAP':
        loader_obj = dl_iemocap.IemocapDataLoader(
            iemocap_path=nldrp.config.IEMOCAP_PATH)
        dataset_dic = loader_obj.data_dict
    elif dataset_name == 'BERLIN':
        loader_obj = dl_berlin.EmodbDataLoader(
            emodb_path=nldrp.config.BERLIN_PATH)
        dataset_dic = loader_obj.data_dict
    else:
        raise NotImplementedError('Dataset: {} is not yet supported '
                                  ''.format(dataset_name))

    return dataset_dic


def opensmile_extract(config_p,
                      wavpath,
                      temp_p):
    extr_command = ['SMILExtract', '-noconsoleoutput',
                    '-C', config_p,
                    '-I', wavpath,
                    '-O', temp_p]



    subprocess.call(extr_command)

    data = arff.loadarff(temp_p)
    feature_vec = np.asarray(data[0].tolist(), dtype=np.float32)

    feature_vec = feature_vec.reshape(-1)[0:-1]
    subprocess.call(['rm', temp_p])
    return feature_vec


def get_features_dic(dataset_dic, config_p):
    features_dic = {}
    total = sum([len(v) for k, v in dataset_dic.items()])
    bar = ChargingBar("Extracting Opensmile Features for {} "
                      "utterances...".format(total), max=total)
    for spkr in dataset_dic:
        features_dic[spkr] = {}
        for id, raw_dic in dataset_dic[spkr].items():
            features_dic[spkr][id] = {}
            # fs = raw_dic['Fs']
            # signal = raw_dic['wav']
            wavpath = raw_dic['wavpath']

            feat_vec = opensmile_extract(config_p,
                              wavpath,
                              '/tmp/opensmile_feats_tmp')
            features_dic[spkr][id]['x'] = feat_vec
            features_dic[spkr][id]['y'] = raw_dic['emotion']

            bar.next()
    bar.finish()
    return features_dic


def safe_mkdirs(path):
    """! Makes recursively all the directory in input path """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            raise IOError(
                ("Failed to create recursive directories: "
                " {}".format(path)
                )
            )


def save_features_dic(opensm_config,
                      features_dic,
                      save_dir):
    utterance_save_dir = os.path.join(save_dir, 'utterance/')
    safe_mkdirs(utterance_save_dir)
    save_p = os.path.join(utterance_save_dir, opensm_config)
    print "Saving Opensmile Features Dictionary in {}".format(save_p)
    joblib.dump(features_dic, save_p, compress=3)
    print "OK!"



def run(dataset,
        save_dir,
        config_p):

        print "Parsing Dataset <{}>...".format(dataset)
        dataset_dic = load_data(dataset)
        print "OK!"

        features_dic = get_features_dic(dataset_dic, config_p)

        opensm_config = dataset+'_linear_emobase2010'
        save_features_dic(opensm_config, features_dic, save_dir)

def get_args():
    """! Command line parser for Opensmile Utterance level feature
    extraction pipeline"""
    parser = argparse.ArgumentParser(
        description='Opensmile Utterance level feature extraction '
                    'pipeline' )
    parser.add_argument("--dataset", type=str,
                        help="""The name of the dataset""",
                        required=True,
                        choices=['SAVEE',
                                 'IEMOCAP',
                                 'BERLIN'])
    parser.add_argument("-o", "--save_dir", type=str,
        help="""Where to store the corresponding binary file full of 
        data that will contain the dictionary for each speaker. 
        Another subdic for all the sentences with their ids  
        and a 1d numpy matrix for each one of them.
        """,
        default=nldrp.config.EXTRACTED_FEATURES_PATH )
    parser.add_argument("--config", type=str,
                        help="""Opensmile configuration PAth""",
                        required=False,
                        default=nldrp.config.OPENSMILE_CONFIG_PATH)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    run(args.dataset,
        args.save_dir,
        args.config)
