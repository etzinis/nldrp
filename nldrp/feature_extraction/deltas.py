"""!
\brief Deltas module for computing the differences from a
given 2d numpy matrix over all rows.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

def compute(features_block):
    """!
    \brief Computes the deltas in between the rows of the given
    features_block

    \param features_block (\a numpy 2D array) 2D numpy block which
    represents the raw features over time that we want to extract
    stats from. (n_samples, n_features)

    \warning Caller function is responsible for checking the validity
    of both parameters.

    \param features_block (\a numpy 2D array) 2D numpy block which
    represents the raw features over time that we want to extract
    stats from. (n_samples-1, n_features) """

    return (features_block[1:]-features_block[0:-1])


def sanity_test():
    import numpy as np

    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=np.float32)
    da = compute(a)

    print a
    print da

if __name__ == "__main__":
    sanity_test()