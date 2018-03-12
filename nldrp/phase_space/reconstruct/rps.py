"""! 
\brief Reconstruct the Phase Space of a Signal

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright National Technical University of Athens
"""

import numpy as np
import os
import sys
nldrp_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, nldrp_dir)
import config
sys.path.insert(0, config.PYUNICORN_PATH)

import pyunicorn.timeseries.recurrence_plot as unicorn_rp

def rps(signal, tau, ed):
    """!
    \brief Given a signal this method reconstructs its phase space
    according to the given parameters (tau--time delay and 
    ed-- embedding dimension)

    \warning the resulting 2D array will have less samples than the 
    given signal in 1D. 

    \returns rps_signal (2D numpy vector) -- phase space representation
    shape = (points in phase space - (ed -1)*tau, 
    embedded dimensions (ed))"""
    phase_space_list = []
    s_len = signal.shape[0]

    for i in np.arange(ed):
        ed_vec = signal[i*tau:s_len-(ed-i-1)*tau]
        phase_space_list.append(ed_vec)

    rps_signal = np.array(phase_space_list)

    return np.transpose(rps_signal)

def dummy_RPS(signal,tau,ed):
    
    rolled_signal = np.roll(signal,-tau)

    phase_space_signal = np.vstack((signal[:-tau],
                                    rolled_signal[:-tau]))
    phase_space_signal = np.transpose(phase_space_signal)

    for dim in range(3,ed+1):
        rolled_signal = np.roll(phase_space_signal[:,-1],
                                -tau)
        rolled_signal= rolled_signal.reshape(-1,1)
        phase_space_signal = np.hstack((
                             phase_space_signal[:-tau],
                             rolled_signal[:-tau]))

    return phase_space_signal


def ami_on_rolled(signal, tau, n_bins):
    signal1 = np.copy(signal)
    rolled_signal = np.roll(signal1, -tau)
    clp_sig = signal1[:-tau]
    clp_rol_sig = rolled_signal[:-tau]
    h2d, _, _ = np.histogram2d(clp_sig, clp_rol_sig, bins=n_bins)
    pxy = h2d / float(np.sum(h2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    ami = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-6) + 1e-6))
    return ami


def ami_tau_estimation(signal, max_tau=100, n_bins=32):
    taus = np.arange(4, max_tau)
    ami_acc = [(ami_on_rolled(signal, x, n_bins), x) for x in np.arange(1, 4)]

    if ami_acc[1][0] < ami_acc[0][0] and ami_acc[1][0] < ami_acc[2][0]:
        return ami_acc[1][1]
    first_ami = ami_acc
    for tau in taus:
        new_ami = ami_on_rolled(signal, tau, n_bins)
        ami_acc[0] = ami_acc[1]
        ami_acc[1] = ami_acc[2]
        ami_acc[2] = (new_ami, tau)
        if ami_acc[1][0] < ami_acc[0][0] and ami_acc[1][0] < ami_acc[2][0]:
            return ami_acc[1][1]

    if first_ami[0][0] < ami_acc[2][0]:
        return first_ami[0][1]
    return ami_acc[2][1]


def cython_RPS(signal, tau, ed):
    """!
    \brief Wrapper of Unicorn Embed Time series static method --
    Cython efficient implementation"""

    return unicorn_rp.RecurrencePlot.embed_time_series(signal,
                                                       ed, tau)


def ami_RPS(signal, max_tau, ed=3):
    """!
    \brief This uses the Average mutual information in order to find
    the optimal time lag and possibly the False Nearest Neighbors for
    determining the optimal embedding dimension."""
    tau = ami_tau_estimation(signal, max_tau=max_tau, n_bins=32)
    return cython_RPS(signal, tau, ed)


def test_performance(iterations=1000):

    import time
    
    f0_list = np.random.uniform(low=40.0, high=700.0, 
                                size=(iterations,))
    f0_list = np.sort(f0_list)
    fs_list = [8000, 16000, 44100]
    win_secs = 0.025

    tau = 1 
    ed = 3

    print '='*5 + ' RPS Performance Testing ' + '='*5
    
    for fs in fs_list:
        total_time = {
            'Numpy Roll':0.0,
            'Python Roll':0.0,
            'Unicorn':0.0,
            'AMI': 0.0
        }
        win_samples = int(win_secs * fs) 
        print '\n\n'+'~'*5 + ' Testing for Fs={} Samples={} '.format(
                    fs, win_samples)+'~'*5
        
        for f0 in f0_list:
            x = np.cos((2.*np.pi * f0 / fs) * np.arange(win_samples))

            before = time.time()
            est_rps_n = dummy_RPS(x, tau, ed)
            now = time.time()
            total_time['Numpy Roll'] += now-before

            before = time.time()
            est_rps_p = rps(x, tau, ed)
            now = time.time()
            total_time['Python Roll'] += now-before

            before = time.time()
            uni_rps = cython_RPS(x, tau, ed)
            now = time.time()
            total_time['Unicorn'] += now - before

            before = time.time()
            ami_rps = ami_RPS(x, 100, ed)
            now = time.time()
            total_time['AMI'] += now - before

            # check the validity of the results 
            assert (abs(est_rps_n-est_rps_p)< 0.00001).all(), (
                'All implementations '
                'of RPS should have the same result')

            # check the validity of the results
            assert (abs(est_rps_n-uni_rps)< 0.00001).all(), (
                'All implementations '
                'of RPS should have the same result')

        for k,v in total_time.items():
            print (">Total Time: {} for {} frames, Method: "
                    "{}".format( v, iterations, k))

def test_rps():
    """!
    \brief Example of usage"""
    dummy_length = 10
    signal = np.array(np.arange(dummy_length))

    l_tau = np.arange(5)
    l_ed = np.arange(5)

    print signal

    for ed in l_ed:
        for tau in l_tau:
            print "Ed: {} Tau: {}".format(ed,tau)
            rps_res = rps(signal, tau, ed)
            print rps_res 

if __name__ == "__main__":
    test_performance(iterations=1000)
    # test_rps()
