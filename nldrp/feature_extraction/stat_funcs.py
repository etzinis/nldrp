"""!
\brief Statistical functionals module for computing the latter from a
given numpy matrix.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew


def compute(features_block):
    """!
    \brief Computes a predefined set of statistical functionals
    for the features_block given.
    \details The selected 18 functionals are self explanatory from the
    code

    \param features_block (\a numpy 2D array) 2D numpy block which
    represents the raw features over time that we want to extract
    stats from. (n_samples, n_features)

    \warning Caller function is responsible for checking the validity
    of both parameters.

    \returns \b np_feat_stats (\a numpy array) Statistics for the
    given feat_col in a 1D array of shape: (18*n_features,) """

    (n_samples, n_features) = features_block.shape
    stats_list = []

    stats_list.append(np.mean(features_block, axis=0))
    stats_list.append(np.var(features_block, axis=0))
    stats_list.append(kurtosis(features_block, axis=0))
    stats_list.append(skew(features_block, axis=0))
    stats_list.append( np.median(features_block, axis=0))

    sorted_block = np.sort(features_block, axis=0)

    stats_list.append(sorted_block[0, :])
    stats_list.append(sorted_block[-1, :])

    selected_perc = [x/100.0 for x in [1,5,25,50,75,95,99]]
    for p in selected_perc:
        perc = sorted_block[int(p*n_samples)]
        stats_list.append(perc)

    ranges = [(big/100.0, small/100.0)
               for (big, small) in
               [(99, 1), (75, 25), (50, 25), (75, 50)]]

    for (big, small) in ranges:
        iqr = sorted_block[int(big*n_samples)] - sorted_block[int(
            small*n_samples)]
        stats_list.append(iqr)

    return np.concatenate(stats_list, axis=0)


def sanity_test():
    n_frames = 3
    n_features = 2
    list_of_features = [np.random.normal(0., 1., n_frames)
                        for x in np.arange(n_features)]

    dummy_block = np.transpose(np.array(list_of_features))

    stats = compute(dummy_block)
    print stats.shape

if __name__ == "__main__":
    sanity_test()