"""!
\brief Opensmile Segment-level extraction by using a specified
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

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)
import nldrp.config
import nldrp.io.dataloader as dl_savee
import nldrp.io.dataloader_emodb as dl_berlin
import nldrp.io.dataloader_iemo as dl_iemocap
import nldrp.feature_extraction.frame_breaker as seg_breaker

from scipy.io.wavfile import write as wav_write

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


def segment_opensmile_extraction(config_p,
                                 segment_signal,
                                 fs,
                                 temp_p):

    temp_seg_path = '/tmp/temp_segment.wav'
    int16_s = np.asarray(segment_signal*32767, dtype=np.int16)
    wav_write(temp_seg_path, fs, int16_s)
    opensmile_feat_vec = opensmile_extract(config_p,
                             temp_seg_path,
                             temp_p)

    subprocess.call(['rm', temp_seg_path])
    return opensmile_feat_vec


def extract_per_segment(config_p,
                        temp_p,
                        segment_dur,
                        segment_ol,
                        fs,
                        signal):

    init_len = len(signal)
    seg_len = int(segment_dur * fs)
    ol_len = int(segment_ol * fs)
    if init_len <= seg_len:
        len_to_pad = seg_len + 1
    else:
        n_segs = int((init_len - seg_len) / ol_len)
        if seg_len + n_segs*ol_len == init_len:
            len_to_pad = seg_len + n_segs*ol_len + 1
        else:
            len_to_pad = seg_len + (n_segs+1) * ol_len + 1
    padded_s = seg_breaker.zero_pad_frame(signal, len_to_pad)

    st_indices, seg_size, seg_step =  \
        seg_breaker.get_frames_start_indices(padded_s,
                                             fs,
                                             segment_dur,
                                             segment_ol)

    segment_feat_vecs = [segment_opensmile_extraction(config_p,
                         signal[st:st+seg_size], fs, temp_p)
                         for st in st_indices]

    all_feat_vecs = np.array(segment_feat_vecs, dtype=np.float32)

    return all_feat_vecs


def get_features_dic(dataset_dic,
                     config_p,
                     segment_dur,
                     segment_ol):
    features_dic = {}
    total = sum([len(v) for k, v in dataset_dic.items()])
    bar = ChargingBar("Extracting Opensmile Features for {} "
                      "utterances...".format(total), max=total)
    for spkr in dataset_dic:
        features_dic[spkr] = {}
        for id, raw_dic in dataset_dic[spkr].items():
            features_dic[spkr][id] = {}

            fs = raw_dic['Fs']
            signal = raw_dic['wav']

            segment_features_2D = extract_per_segment(
                                  config_p,
                                  '/tmp/opensmile_feats_tmp',
                                  segment_dur,
                                  segment_ol,
                                  fs,
                                  signal)

            features_dic[spkr][id]['x'] = segment_features_2D
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
    utterance_save_dir = os.path.join(save_dir, 'segment/')
    safe_mkdirs(utterance_save_dir)
    save_p = os.path.join(utterance_save_dir, opensm_config)
    print "Saving Opensmile Features Dictionary in {}".format(save_p)
    joblib.dump(features_dic, save_p, compress=3)
    print "OK!"



def run(dataset,
        save_dir,
        config_p,
        segment_dur,
        segment_ol):

        print "Parsing Dataset <{}>...".format(dataset)
        dataset_dic = load_data(dataset)
        print "OK!"

        features_dic = get_features_dic(dataset_dic,
                                        config_p,
                                        segment_dur,
                                        segment_ol)

        opensm_config = dataset+('_linear_emobase2010_segl_{}_segol_'
                                 '{}'.format(segment_dur, segment_ol))
        save_features_dic(opensm_config, features_dic, save_dir)


def get_args():
    """! Command line parser for Opensmile Segment level feature
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
        default=nldrp.config.EXTRACTED_FEATURES_PATH)
    parser.add_argument("--config", type=str,
                        help="""Opensmile configuration PAth""",
                        required=False,
                        default=nldrp.config.OPENSMILE_CONFIG_PATH)
    parser.add_argument("--segment_dur", type=float,
                        help="""The specified length of the segments 
                        in seconds
                        """,
                        required=False,
                        default=1.0)
    parser.add_argument("--segment_ol", type=float,
                        help="""The specified overlap in seconds 
                        between the 
                        segments""",
                        required=False,
                        default=0.5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    run(args.dataset,
        args.save_dir,
        args.config,
        args.segment_dur,
        args.segment_ol)
