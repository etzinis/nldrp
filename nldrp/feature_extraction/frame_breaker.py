"""! 
@brief Apply Window module 
@details This module has the purpose of returning a list of frames
from a given signal. Based on a certain configuration for frame length
and frame step.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright The author
also works for Behavioral Signals which is the only company having
full rights upon both commercial and non-commercial use of this code.
"""

import numpy as np 


def zero_pad_frame(frame, size_to_pad):
    """! Pads with zeros the given frame vector up to the size_to_pad 
    if is bigger than the given size_to_pad then returns the first 
    <size_to_pad> elements from the vector frame.  

    \param frame (\a numpy array) A frame of audio
    \param size_to_pad (\a int) The length that the given frame should
     be zero padded to. 
    
    \throws TypeError If it cannot pad the input frame because it was 
    trying to pad an array with zeros from an input signal with no 
    numerical values
    
    \returns \b zero_frame (\a numpy array) The zero padded or 
    truncated frame vector from the corresponding given frame"""

    try:
        frame_len = len(frame)
        if frame_len >= size_to_pad:
            return frame[:size_to_pad]
        zero_frame = np.zeros(size_to_pad)
        zero_frame[:frame_len]=frame
        return zero_frame
    except Exception as e:
        print (("Cannot zero pad the input frame"
            " Expected: 1D numpy matrix with len()>0"
            ))
        raise e

def convert_seconds_to_samples(frame_duration,
                               frame_stride, 
                               sample_rate):
    """! Converts the given values: (frame_duration, frame_stride) 
    in their corresponding (frame_size, frame_step) which are measured 
    in numbers of samples of a signal with Fs=sample_rate

    \param frame_duration (\a int) The duration (in seconds) that 
    every frame lasts.
    \param frame_stride (\a int) The duration (in seconds) that every
    next frame begins. (How far the next frame should start)  
    \param sample_rate (\a int) The sample frequency of the audio 
    signal from which the given frame/segment is given
    
    \throws TypeError If it is given non numeric values as input
    
    \returns \b frame_size, frame_step (\a int, int) The corresponding
    durations given in seconds expressed as number of samples for a 
    signal with the given sample_rate (==sample frequency) """

    try:
        frame_size = int(frame_duration * sample_rate)
        frame_step = int(frame_stride * sample_rate)
        return frame_size, frame_step
    except Exception as e:
        print ("Could not convert user defined values to number of "
        "samples. frame_duration = {}, frame_stride = {}, "
        "fs = {}".format(frame_duration, frame_stride, sample_rate))
        raise e

def get_frames(signal, 
               fs, 
               frame_duration=0.025, 
               frame_stride=0.01):
    """! Breaks the given signal to its corresponding frames according 
    to the given values: (frame_duration, frame_stride, fs). This 
    function returns all the features in a 2D numpy array in which: 
    1) every row corresponds to a frame 2) every column corresponds to 
    a sample from the intial signal and belongs to this frame.

    \waring This implementation does not take into account the last 
    frame. This could raise a little bit code complexity and it has 
    not a practical use (default value is ~25ms -- no significant 
    information can be hidden there). Its also users responsibility 
    that all the given values are rational e.g. len(signal) > 
    frame_duration. In this case [] will be returned -- no frames.  

    \param signal (\a numpy array) The given audio signal that has to 
    be breaken into frames 
    \param fs (\a int) The sample frequency of the audio signal 
    \param frame_duration (\a int) The duration (in seconds) that 
    every frame lasts.
    \param frame_stride (\a int) The duration (in seconds) that every
    next frame begins. (How far the next frame should start)  
    
    \throws TypeError If it cannot break the signal to its frames 
    because the input frame because does not contain numerical values
    
    \returns \b frames_2D (\a 2D numpy array) The corresponding
    frame vector with shape: (number_of_frames,
     number_of_sample_per_frame)."""
    
    frame_size, frame_step = convert_seconds_to_samples(
                    frame_duration, frame_stride, fs)

    try:
        sample_size = len(signal)
        return np.array([signal[x:x+frame_size] 
            for x in np.arange(0,sample_size-frame_size,frame_step)])
    except Exception as e:
        print ("Could not return 2D numpy matrix with all the frames")
        raise e
    

def get_frames_start_indices(signal, 
                            fs, 
                            frame_duration=0.025, 
                            frame_stride=0.01):
    """! Breaks the given signal to its corresponding frames according 
    to the given values: (frame_duration, frame_stride, fs). This 
    function returns all the starting indices of the corresponding 
    frames in an 1D numpy array. This function should be used if a 
    caller function wants to perform late evaluation for optimization, 
    parallelization, etc. In order not to construct the full matrix 
    of the frames but only the values that are needed. It also returns 
    the frame_size and frame_step because they might be needed from 
    the caller function in order to iterate through the frames.

    \warning This implementation does not take into account the last 
    frame. This could raise a little bit code complexity and it has 
    not a practical use (default value is ~25ms -- no significant 
    information can be hidden there). Its also users responsibility 
    that all the given values are rational e.g. len(signal) > 
    frame_duration. In this case [] will be returned -- no frames.  

    \param signal (\a numpy array) The given audio signal that has to 
    be breaken into frames 
    \param fs (\a int) The sample frequency of the audio signal
    \param frame_duration (\a int) The duration (in seconds) that 
    every frame lasts.
    \param frame_stride (\a int) The duration (in seconds) that every
    next frame begins. (How far the next frame should start)  
    
    \throws TypeError If it cannot find the indices of the frames of
    the given signal because of Type misallignment

    \returns (indices, frame_size, frame_step)
    \returns \b indices (\a 1D numpy array) The corresponding
    frame start indices that can be later used for iterating or 
    extracting the frames from the caller function. Shape = (
    Number of Frames,)
    \returns \b frame_size, frame_step (\a int, int) The corresponding
    durations given in seconds expressed as number of samples for a 
    signal with the given sample_rate (==sample frequency)
    """
   
    frame_size, frame_step = convert_seconds_to_samples(
                    frame_duration, frame_stride, fs)
    try:
        sample_size = len(signal)
        return (np.arange(0,sample_size-frame_size,frame_step), 
                frame_size, frame_step)
    except Exception as e:
        print ("Could not return numpy array of indices corresponding "
            "to the beginnings of the frames for the given signal.")
        raise e


if __name__ == '__main__':
    frame = np.array([0,1,2,3,4,5,6,7,8,9])
    padded_frame = zero_pad_frame(frame, 20)
    # print padded_signal

    s_len = 800
    fs = 16000
    signal = np.random.normal([1.0*x for x in range(s_len)])

    frames = get_frames(signal, fs, 
        frame_duration=0.025, frame_stride=0.01)

    print "Frames in 2D matrix of shape: {}".format(frames.shape)

    indices, frame_size, frame_step = get_frames_start_indices(signal, 
        fs, frame_duration=0.025, frame_stride=0.01)
    print ("Start Indices in Numpy Array: {} Frame Size (samples): {}"
        " Frame Step Size (in Samples): {}"
        "".format(indices,frame_size, frame_step))