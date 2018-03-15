"""!
\brief Utterance level feature pipeline for stats over rqa measures
fully configurable

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import argparse
import itertools
import utterance_feat_extraction as feat_extract


def generate_grid_space(param_grid):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in
                   itertools.product(*values)]
    return experiments


def extract_all(args):

    param_grid = {
                 'dataset':[args.dataset],
                 'cache_dir': [args.cache_dir],
                 'save_dir':[args.save_dir],
                 'phase_space_method':args.tau_est_methods,
                 'time_lag':args.taus,
                 'embedding_dimension':[3],
                 'norm':args.norms,
                 'thresh_method':args.thresh_methods,
                 'thresh':args.threshs,
                 'l_min':[2],
                 'v_min':[2],
                 'w_min':[1],
                 'frame_duration':args.frame_durations,
                 # 'frame_stride':args.frame_durations / 2.0
            }

    all_configs = generate_grid_space(param_grid)
    for i, config_dic in enumerate(all_configs):
        print "Extracting {}/{}...".format(i+1, len(all_configs))
        config_dic['frame_stride'] = config_dic['frame_duration'] / 2.0
        # from pprint import pprint
        # pprint(config_dic)

        try:
            feat_extract.run(config_dic)
        except Exception as e:
            print(e)
            exit()


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
    parser.add_argument("--save_dir", type=str,
        help="""Where to store the corresponding binary file full of 
        data that will contain the dictionary for each speaker. 
        Another subdic for all the sentences with their ids  
        and a 1d numpy matrix for each one of them.
        """,
        required=True )
    parser.add_argument("--taus", type=int, nargs='+',
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
