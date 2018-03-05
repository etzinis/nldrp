"""!
\brief Statistical functionals module for computing the latter from a
given numpy matrix.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import numpy as np

def compute(features_block):
    """!
    \brief Computes a predefined set of statistical functionals
    for the features_block given.

    \param features_block (\a numpy 2D array) 2D numpy block which
    represents the raw features over time that we want to extract
    stats from. (n_samples, n_short_term_features)

    \warning Caller function is responsible for checking the validity
    of both parameters.

    \returns \b np_feat_stats (\a numpy array) Statistics for the
    given feat_col in a 1D array of shape: (len(selected_stats),) """