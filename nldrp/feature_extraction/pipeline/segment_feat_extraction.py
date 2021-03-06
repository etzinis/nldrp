"""!
\brief Segment level feature pipeline for stats over rqa measures
from the frame level analysis and after that combining them on
utterance level by using various statistical functionals.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import os
import sys
from sklearn.externals import joblib
from progress.bar import ChargingBar
import time
import pprint
import numpy as np

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config
import nldrp.recurrence_plots.rqa.seg_stats_rqa as rqa_stats

import nldrp.io.dataloader as dl_savee
import nldrp.io.dataloader_emodb as dl_berlin
import nldrp.io.dataloader_iemo as dl_iemocap
import nldrp.feature_extraction.frame_breaker as seg_breaker


def load_dataset_and_cache(dataset_name,
                           cache_dir):

    cache_path = os.path.join(cache_dir, dataset_name+'_datadic.bin')
    if os.path.lexists(cache_path):
        print "Loading from cache..."
        dataset_dic = joblib.load(cache_path)
        return dataset_dic

    print "Loading from dataloader..."
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
        raise NotImplementedError('Dataset: {} is not yet integrated '
                            'in this pipeline'.format(dataset_name))

    print "Caching this dataset dictionary..."
    joblib.dump(dataset_dic, cache_path, compress=3)
    return dataset_dic


def rqa_feats_for_this_segment(fs,
                               config,
                               signal):
    seg_extr = rqa_stats.SegmentRQAStatistics(fs=fs, **config)
    return seg_extr.extract(signal)


def extract_per_segment(fs,
                        config,
                        signal,
                        segment_dur,
                        segment_ol):
    init_len = len(signal)
    seg_len = int(segment_dur * fs)
    ol_len = int(segment_ol * fs)
    if init_len <= seg_len:
        len_to_pad = seg_len + 1
    else:
        n_segs = int((init_len - seg_len) / ol_len)
        if seg_len + n_segs * ol_len == init_len:
            len_to_pad = seg_len + n_segs * ol_len + 1
        else:
            len_to_pad = seg_len + (n_segs + 1) * ol_len + 1

    padded_s = seg_breaker.zero_pad_frame(signal, len_to_pad)

    st_indices, seg_size, seg_step = \
        seg_breaker.get_frames_start_indices(padded_s,
                                             fs,
                                             segment_dur,
                                             segment_ol)

    segment_feat_vecs = [rqa_feats_for_this_segment(
                         fs, config, signal[st:st + seg_size])
                         for st in st_indices]

    all_feat_vecs = np.array(segment_feat_vecs, dtype=np.float32)

    return all_feat_vecs


def get_features_dic(dataset_dic,
                     config,
                     segment_dur,
                     segment_ol):

    features_dic = {}
    total = sum([len(v) for k, v in dataset_dic.items()])
    bar = ChargingBar("Extracting RQA Measures for {} "
                      "utterances...".format(total), max=total)
    for spkr in dataset_dic:
        features_dic[spkr] = {}
        for id, raw_dic in dataset_dic[spkr].items():
            features_dic[spkr][id] = {}
            fs = raw_dic['Fs']
            signal = raw_dic['wav']

            segment_features_2D = extract_per_segment(fs,
                                                      config,
                                                      signal,
                                                      segment_dur,
                                                      segment_ol)

            features_dic[spkr][id]['x'] = segment_features_2D
            features_dic[spkr][id]['y'] = raw_dic['emotion']

            bar.next()
    bar.finish()
    return features_dic, fs

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


def save_feature_dic(feature_dic,
                     config,
                     fs,
                     seg_dur,
                     seg_ol):

    exper_dat_name = ('{}-rqa-{}-tau-{}-{}-{}-{}-dur-{}-fs-{}-segd-'
                      '{}-segstr-{}'
                      '.dat'.format(
                        config['dataset'],
                        config['phase_space_method'],
                        config['time_lag'],
                        config['norm'],
                        config['thresh_method'],
                        config['thresh'],
                        config['frame_duration'],
                        fs,
                        seg_dur,
                        seg_ol
                      ))

    seg_save_dir = os.path.join(config['save_dir'], 'segment/')
    safe_mkdirs(seg_save_dir)
    save_p = os.path.join(seg_save_dir, exper_dat_name)
    print "Saving Features Dictionary in {}".format(save_p)
    joblib.dump(feature_dic, save_p, compress=3)
    print "OK!"


def features_are_already_extracted(config, fs):
    exper_dat_name = ('{}-rqa-{}-tau-{}-{}-{}-{}-dur-{}-fs-{}'
                      '.dat'.format(
        config['dataset'],
        config['phase_space_method'],
        config['time_lag'],
        config['norm'],
        config['thresh_method'],
        config['thresh'],
        config['frame_duration'],
        fs
    ))

    seg_save_dir = os.path.join(config['save_dir'], 'segment/')
    save_p = os.path.join(seg_save_dir, exper_dat_name)
    if os.path.lexists(save_p):
        print "Found features in: {}".format(save_p)
        return True
    return False


def run(config,
        segment_dur,
        segment_ol):

    print "Parsing Dataset <{}>...".format(config['dataset'])
    dataset_dic = load_dataset_and_cache(config['dataset'],
                                         config['cache_dir'])
    print "OK!"

    freq = {}
    for spkr in dataset_dic:
        for id, raw_dic in dataset_dic[spkr].items():
            try:
                freq[raw_dic['emotion']] += 1
            except:
                freq[raw_dic['emotion']] = 1
    print freq

    fs = None
    for spkr in dataset_dic:
        for id, raw_dic in dataset_dic[spkr].items():
            fs = raw_dic['Fs']
            break
        break

    if features_are_already_extracted(config, fs):
        return

    before = time.time()
    features_dic, fs = get_features_dic(dataset_dic,
                                        config,
                                        segment_dur,
                                        segment_ol
                                        )

    now = time.time()
    print "Finished Extraction after: {} seconds!".format(
         time.strftime('%H:%M:%S', time.gmtime(now - before)))

    save_feature_dic(features_dic,
                     config,
                     fs,
                     segment_dur,
                     segment_ol)


def get_args():
    """! Command line parser for Utterance level feature extraction
    pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level feature extraction pipeline' )
    parser.add_argument("--cache_dir", type=str,
        help="""Directory which would be available to store some 
        binary files for quicker load of dataset""",
        default='/tmp/')
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
    parser.add_argument("--tau", type=int,
                        help="""Time Delay Ad-hoc""",
                        default=1)
    parser.add_argument("--tau_est_method", type=str,
                        help="""How to estimate Time Delay (Using 
                        an adhoc value as set or estimate AMI per 
                        frame?)""",
                        default='ad_hoc',
                        choices=['ad_hoc', 'ami'])
    parser.add_argument("--norm", type=str,
                        help="""Norm for computing in RPs""",
                        default='euclidean',
                        choices=["manhattan", "euclidean", "supremum"])
    parser.add_argument("--thresh_method", type=str,
                        help="""How to threshold RPs""",
                        default='threshold',
                        choices=["threshold",
                                "threshold_std",
                                "recurrence_rate"])
    parser.add_argument("--thresh", type=float,
                        help="""Value of threshold in (0,1)""",
                        default=0.1)
    parser.add_argument("--frame_duration", type=float,
                        help="""Frame duration in seconds""",
                        default=0.02)
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
    config = {
        'dataset':args.dataset,
        'cache_dir':args.cache_dir,
        'save_dir':args.save_dir,
        'phase_space_method':args.tau_est_method,
        'time_lag':args.tau,
        'embedding_dimension':3,
        'norm':args.norm,
        'thresh_method':args.thresh_method,
        'thresh':args.thresh,
        'l_min':2,
        'v_min':2,
        'w_min':1,
        'frame_duration':args.frame_duration,
        'frame_stride':args.frame_duration / 2.0
    }
    run(config,
        args.segment_dur,
        args.segment_ol)
