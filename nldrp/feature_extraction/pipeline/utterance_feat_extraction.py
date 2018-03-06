"""!
\brief Utterance level feature pipeline for stats over rqa measures
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

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config
import nldrp.recurrence_plots.rqa.seg_stats_rqa as rqa_stats


def load_dataset_and_cache(dataset_name,
                           cache_dir):

    cache_path = os.path.join(cache_dir, dataset_name+'_datadic.bin')
    if os.path.lexists(cache_path):
        print "Loading from cache..."
        dataset_dic = joblib.load(cache_path)
        return dataset_dic

    import nldrp.io.dataloader as dataloader

    print "Loading from dataloader..."
    if dataset_name == 'SAVEE':
        loader_obj = dataloader.SaveeDataloader(
                      savee_path=nldrp.config.SAVEE_PATH)
        dataset_dic = loader_obj.data_dict
    else:
        raise NotImplementedError('Dataset: {} is not yet integrated '
                            'in this pipeline'.format(dataset_name))

    print "Caching this dataset dictionary..."
    joblib.dump(dataset_dic, cache_path, compress=3)
    return dataset_dic


def get_features_dic(dataset_dic):
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

            seg_extr = rqa_stats.SegmentRQAStatistics(fs=fs, **config)
            features_dic[spkr][id]['x'] = seg_extr.extract(signal)
            features_dic[spkr][id]['y'] = raw_dic['emotion']

            bar.next()
    bar.finish()
    return features_dic


# def save_feature_dic(feature_dic):



def run(config):

    print "Parsing Dataset <{}>...".format(config['dataset'])
    dataset_dic = load_dataset_and_cache(config['dataset'],
                                         config['cache_dir'])
    print "OK!"

    before = time.time()
    features_dic = get_features_dic(dataset_dic)
    now = time.time()
    print "Finished Extraction after: {} seconds!".format(
         time.strftime('%H:%M:%S', time.gmtime(now - before)))

    save_p = 'dsf'
    print "Saving Features Dictionary in {}".format(save_p)

def get_args():
    """! Command line parser for Utterance level feature pipeline"""
    parser = argparse.ArgumentParser(
        description='Utterance level feature pipeline' )
    parser.add_argument("--cache_dir", type=str,
        help="""Directory which would be available to store some 
        binary files for quicker load of dataset""",
        default='/tmp/')
    parser.add_argument("--dataset", type=str,
                        help="""The name of the dataset""",
                        required=True,
                        choices=['SAVEE'])
    parser.add_argument("-o", "--save_dir", type=str,
        help="""Where to store the corresponding binary file full of 
        data that will contain the dictionary for each speaker. 
        Another subdic for all the sentences with their ids  
        and a 1d numpy matrix for each one of them.
        """,
        default='/raid/processing/talkiq/stereo_converted_recordings' )
    parser.add_argument("-tau", type=int,
                        help="""Time Delay Ad-hoc""",
                        default=1)
    parser.add_argument("--tau_est_method", type=str,
                        help="""How to estimate Time Delay (Using 
                        an adhoc value as set or estimate AMI per 
                        frame?)""",
                        default='ad_hoc',
                        choices=['ad_hoc', 'ami'])
    parser.add_argument("-norm", type=str,
                        help="""Norm for computing in RPs""",
                        default='euclidean',
                        choices=["manhattan", "euclidean", "supremum"])
    parser.add_argument("--thresh_method", type=str,
                        help="""How to threshold RPs""",
                        default='threshold',
                        choices=["threshold",
                                "threshold_std",
                                "recurrence_rate"])
    parser.add_argument("-thresh", type=float,
                        help="""Value of threshold in (0,1)""",
                        default=0.1)
    parser.add_argument("--frame_duration", type=float,
                        help="""Frame duration in seconds""",
                        default=0.02)
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
    run(config)