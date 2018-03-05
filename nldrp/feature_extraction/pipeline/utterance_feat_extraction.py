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

nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, nldrp_dir)

import nldrp.config as config


def load_dataset_and_cache(dataset_name,
                           cache_dir)

    cache_path = os.path.join(cache_dir, dataset_name)
    if os.path.lexists(cache_path):


def run():
    pass


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()