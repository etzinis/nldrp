"""! 
\brief Reconstruct the Phase Space of a Signal

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import numpy as np

class RP(object):
    """Computation of various RPs"""
    def __init__(self, 
                 RP_name, 
                 thresh=0.15,
                 valid_RP_names = ['binary', 'continuous']):
        """!
        \brief This constructor checks whether all the parameters 
        have been set appropriatelly and finally set them. 
        Essentially we get an abstraction for the RP extraction 
        as the extractor is set internally."""
        
        if RP_name in valid_RP_names:
            self.RP_name = RP_name
        else:
            raise NotImplementedError(('Recurrence Plot Name: <{}> '
                'is not a valid RP class name. Please use one of the '
                'following: {}'.format(RP_name, valid_RP_names)))

        if thresh > 0.0 and thresh < 1.0:
            self.thresh = thresh
        else:
            raise ValueError('Threshold <{}> not set into (0,1)'
                ''.format(thresh))

if __name__ == "__main__":
    rp_constructor = RP('continuous')