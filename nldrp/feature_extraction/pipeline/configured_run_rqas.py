"""!
\brief Utterance level feature pipeline for stats over rqa measures
fully configurable

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import 
import utterance_feat_extraction as feat_ectract


def generate_grid_space(self):
    keys, values = zip(*self.param_grid.items())
    experiments = [dict(zip(keys, v)) for v in
                   itertools.product(*values)]
    fold_gens = itertools.tee(self.folds_gen, len(experiments))
    for i, v in enumerate(experiments):
        yield v, fold_gens[i]


def extract_all(args):
    print args


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
        required=True )
    parser.add_argument("-taus", type=int, nargs='+',
                        help="""Time Delay Ad-hoc""",
                        default=[1])
    parser.add_argument("--tau_est_methods", type=str, nargs='+',
                        help="""How to estimate Time Delay (Using 
                        an adhoc value as set or estimate AMI per 
                        frame?)""",
                        default=['ad_hoc'],
                        choices=['ad_hoc', 'ami'])
    parser.add_argument("--norms", type=str,
                        help="""Norm for computing in RPs""", nargs='+',
                        default=['euclidean'],
                        choices=["manhattan", "euclidean", "supremum"])
    parser.add_argument("--thresh_methods", type=str, nargs='+',
                        help="""How to threshold RPs""",
                        default=['threshold'],
                        choices=["threshold",
                                "threshold_std",
                                "recurrence_rate"])
    parser.add_argument("--threshs", type=float, nargs='+',
                        help="""Value of threshold in (0,1)""",
                        default=[0.1])
    parser.add_argument("--frame_durations", type=float, nargs='+',
                        help="""Frame duration in seconds""",
                        default=[0.02])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    extract_all(args)
