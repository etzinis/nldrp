"""!
\brief Segment Level Features Fusion

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import numpy as np
from sklearn.externals import joblib
import os


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
                    assert el_dic['y'] == final_data_dic[spkr][id][
                        'y'], 'Both ids should have the same emotion ' \
                              'label'
                    prev_vec = final_data_dic[spkr][id]['x']
                    this_vec = el_dic['x']

                    assert prev_vec.shape[0] == this_vec.shape[0], \
                        'The two arrays to be concatenated should ' \
                        'contain the same number of segments'

                    new_vec = np.concatenate([prev_vec, this_vec],
                                             axis=1)

                    final_data_dic[spkr][id]['x'] = new_vec
        except Exception as e:
            print "Failed to update the Fused dictionary"
            raise e

    return final_data_dic


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


def save_dictionary(fused_dic,
                    save_p):

    dir_path = os.path.dirname(save_p)
    safe_mkdirs(dir_path)
    print "Saving Fused per Segment Features Dictionary in {}".format(
        save_p)
    joblib.dump(fused_dic, save_p, compress=3)
    print "OK!"


def get_args():
    """! Command line parser for Utterance level classification Leave
    one speaker out schema pipeline -- Find Best Models"""
    parser = argparse.ArgumentParser(
        description='Command line fuser of segment extracted features '
                    'in joblib loadable format')
    parser.add_argument('-i', '--input_features_paths', nargs='+',
                        help='File paths of the features you want to '
                             'concatenate their segment features',
                        required=True)
    parser.add_argument('-o', '--output_path', type=str,
                        help='Path where to store the fused dictionary',
                        required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    fused_dic = fuse_all_configurations(args.input_features_paths)
    save_dictionary(fused_dic, args.output_path)