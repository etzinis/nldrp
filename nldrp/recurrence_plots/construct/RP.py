"""! 
\brief Reconstruct the Phase Space of a Signal
\details RP extractors in Factory class format for better abstraction

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import numpy as np

class RP(object):
    """Computation of various RPs"""

    @staticmethod
    def factory(RP_name, 
                normalized=True,
                thresh=0.15):
        """!
        \brief This constructor checks whether all the parameters 
        have been set appropriatelly and finally set them by also 
        returning the appropriate implementation class. 
        Essentially we get an abstraction for the RP extraction 
        as the extractor is set internally."""

        if RP_name == 'continuous':
            return ContinuousRP(thresh)
        elif RP_name == 'binary':
            return BinaryRP(normalized)
        else:
            valid_RP_names = ['binary', 'continuous']
            raise NotImplementedError(('Recurrence Plot Name: <{}> '
                'is not a valid RP class name. Please use one of the '
                'following: {}'.format(RP_name, valid_RP_names)))

class ContinuousRP(RP):
    """docstring for ContinuousRP"""
    def __init__(self, thresh):
        if thresh > 0.0 and thresh < 1.0:
            self.thresh = thresh
        else:
            raise ValueError('Threshold <{}> not set into (0,1)'
                ''.format(thresh))

    def extract_RP(self):
        pass
        
class BinaryRP(object):
    """docstring for BinaryRP"""
    def __init__(self, normalized):
        self.normalized = normalized

    def extract_RP(self):
        pass

if __name__ == "__main__":
    rp_constr = RP.factory('continuous')
    print rp_constr