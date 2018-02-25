"""! 
\brief Compute Distance matrices for Recurrence Plots

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

from sklearn.metrics.pairwise import pairwise_distances

def compute_distance_matrix(X,
                            norm='euclidean',
                            n_jobs = 1
                            ):
    """!
    \brief """
    return pairwise_distances(X, Y=X, metric=norm, n_jobs=n_jobs)


if __name__ == "__main__":
    import numpy as np
    X = np.array([[1.,2.],[-2.,0.],[0.,0.]])
    D = compute_distance_matrix(X,
                            norm='euclidean',
                            n_jobs = 1
                            )
    print D