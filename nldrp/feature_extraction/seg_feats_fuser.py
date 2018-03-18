"""!
\brief Segment Level Features Fusion

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import numpy as np
from sklearn.externals import joblib
import copy


def fuse_all_configurations(list_of_paths):


    try:
        list_of_dics = [joblib.load(p) for p in list_of_paths]
    except Exception as e:
        print "At least one file path is required"
        raise e

    final_data_dic = list_of_dics.pop(0)

    while list_of_dics:
        temp_dic = list_of_dics.pop(0)
        try:
            for spkr in temp_dic:
                for id, el_dic in temp_dic[spkr].items():
                    assert el_dic['y'] == final_data_dic[spkr][id]['y']
                    prev_vec = list(final_data_dic[spkr][id]['x'])
                    this_vec = list(el_dic['x'])
                    new_vec = np.array(prev_vec+this_vec,
                                       dtype=np.float32)

                    final_data_dic[spkr][id]['x'] = new_vec
        except Exception as e:
            print "Failed to update the Fused dictionary"
            raise e

    return final_data_dic


def get_args():
    """! Command line parser for Utterance level classification Leave
    one speaker out schema pipeline -- Find Best Models"""
    parser = argparse.ArgumentParser(
        description='Command line fuser of segment extracted features '
                    'in joblib loadable format')
    parser.add_argument('-i', '--input_features_paths', nargs='+',
                        help='File paths of the features you want to '
                             'concatenate their segment features')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    fuse_all_configurations(args.input_features_paths)