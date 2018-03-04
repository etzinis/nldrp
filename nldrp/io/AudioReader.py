"""! 
@brief Audio Reader module 
@details This module has the purpose of loading/reading various types
of audio formats and returning a vector and other information that can 
be used from other modules in any pipeline.

@author Efthymios Tzinis {etzinis@gmail.com}
@copyright The author
also works for Behavioral Signals which is the only company having
full rights upon both commercial and non-commercial use of this code.
"""

import argparse
import os
import pydub
import numpy as np


class AudioFile(object):
    """! This class has the purpose of abstracting information
    about parsing audio files, getting their numpy representations, 
    manipulate numpy representations for normalizing them 
    and also some other properties (fs, duration, sample width, etc.)
    """

    def __init__(self, path, suffix=None):
        """! Object Constructor of AudioFile Class
        @param path (@a str) Path of the audio file
        @param suffix (\a str) the extension of the file e.g. '.wav'
        
        \warning suffix is automatically inferred if it is not
        explicitly defined as a parameter

        \throws IOError If pydub fails to read the file 
        
        \returns \b AudioFile object
        """
        self.path = path
        if suffix is not None:
            self.suffix = suffix
        else:
            self.suffix = path.split(".")[-1]
        self.audio = self.read_audio_with_pydub()

    def read_audio_file(self, normalize=True, mix_channels=False):
        """! Wrapper for AudioFile Class file Reader
        @details You have to use this method for reading audio files 
        instead you know what you are doing. This is a level of 
        abstraction for reading th files in channels, normalized or
        not, mixing all the channels together or not.

        @param normalize (@a boolean) Flag for configuring the 
        return signal format. If normalize=True then there will be a
        [-1.0,1.0] normalization. Otherwise, full integer values 
        will be returned. 
        @param mix_channels (@a boolean) Flag for configuring the 
        return signal format. If mix_channels=True then the return 
        value will be a single numpy array corresponding to the mixed
        audio signal (or either the audio signal has only one
        channel). Otherwise, a list of those arrays will be returned.  

        \throws Errors from get_norm_mixed_channel_numpy_array()
        get_norm_numpy_array_in_separated_channels()
        get_mixed_channel_numpy_array()
        get_numpy_array_in_separated_channels()
        
        \returns (\a numpy array of floats) If normalized=True and 
        (mix_channels=True or the audio has only one channel
        \returns (\a numpy array of ints) If normalized=False and 
        (mix_channels=True or the audio has only one channel 
        \returns (\a list of numpy array of floats) If normalized=True 
        and (mix_channels=False and self.audio.get_n_channels() > 1)
        \returns (\a list of numpy array of ints) If normalized=False 
        and (mix_channels=False and self.audio.get_n_channels() > 1)
        """
        if normalize and mix_channels:
            return self.get_norm_mixed_channel_numpy_array()
        elif normalize and not mix_channels:
            return self.get_norm_numpy_array_in_separated_channels()
        elif not normalize and mix_channels:
            return self.get_mixed_channel_numpy_array()
        else:
            return self.get_numpy_array_in_separated_channels()

    def read_audio_with_pydub(self):
        """! Pydub loader wrapper 
        
        \throws IOError If pydub fails to read the file
        
        \returns \b audio (\a pydub.audio_segment.AudioSegment
        object)"""
        try:
            return pydub.AudioSegment.from_file(self.path,
                                                format=self.suffix)
        except Exception as e:
            raise IOError("Path: {} was not found or suffix: <{}> is "
                          " not compatible with the file formats that PyDub can"
                          " support".format(self.path, self.suffix))

    def get_n_channels(self):
        """! Get number of channels for the loaded audiofile 
        \returns \b self.audio.channels (\a int) """
        return self.audio.channels

    def get_fs(self):
        """! Get Sampling frequency for the loaded audiofile 
        \returns \b self.audio.frame_rate (\a int) in Hz"""
        return self.audio.frame_rate

    def get_sample_width(self):
        """! Get Sampling Width for the loaded audiofile 
        \returns \b self.audio.sample_width (\a int) in bytes"""
        return self.audio.sample_width

    def get_duration_seconds(self):
        """! Get Duration of the loaded audiofile 
        \returns \b self.audio.duration_seconds (\a float) in secs"""
        return self.audio.duration_seconds

    def get_pydub_audio_in_mono_channels(self):
        """! Split pydub audio to separate mono channels
        
        \throws RuntimeError If pydub fails to split channels
        
        \returns (\a list of pydub.audio_segment.AudioSegment
        object) Each one is a separate channel"""
        try:
            return self.audio.split_to_mono()
        except Exception as e:
            raise RuntimeError("Could not split to mono channels")

    @staticmethod
    def convert_pydub_audio_to_numpy(pydub_audio):
        """! Convert a pydub audio to a numpy corresponding array 
        
        \param pydub_audio (\a pydub.audio_segment.AudioSegment) 
        A loaded audiofile in pydub format
        
        \throws RuntimeError If could not convert to a numpy array
        
        \returns \b samples (\a numpy array) 1D numpy array with
        size = (number_of_samples,)"""
        try:
            samples = pydub_audio.get_array_of_samples()
            return np.array(samples)
        except Exception as e:
            raise RuntimeError("Could not return numpy samples array")

    def get_numpy_array_from_pydub_audio(self):
        """! Return a numpy corresponding array to saved pydub audio 
        
        \throws RuntimeError If could not convert the saved pydub audio
        data to a numpy array
        
        \returns (\a int numpy array) 1D numpy array with
        size = (number_of_samples,)
        
        \warning If n_channels > 1 then the resulting numpy array 
        corresponds to channel_1_np_array | channel_2_np_array | ...
        concatenated. This should be not used externally, except if
        someone knows what he is doing. Better use: 
        get_numpy_array_in_separated_channels()"""
        return self.convert_pydub_audio_to_numpy(self.audio)

    def get_numpy_array_in_separated_channels(self):
        """! Get a list of numpy corresponding arrays for all the 
        available channels, form saved pydub audio 
        
        \throws RuntimeError If could not convert the saved pydub audio
        data to a numpy array (in any of the channels)
        
        \returns (\a list of int numpy arrays) List of 1D numpy arrays 
        with size = (number_of_samples,), each one of them"""
        separated_ch = self.get_pydub_audio_in_mono_channels()
        numpy_separated_ch = map(lambda x:
                                 self.convert_pydub_audio_to_numpy(x),
                                 separated_ch)
        return numpy_separated_ch

    @staticmethod
    def normalize_audio(audio_vector, sample_width):
        """! Normalize the a signal vector considering the sample width
        in bytes that are given as input.
        
        \param audio_vector (\a numpy array 1D) Signal in numpy array
        \param sample_width (\a int) Number of bytes per sample in 
        order to infer the maximum amplitude of integer input array
        
        \throws Exception On failure to normalize input numpy array
        
        \returns \b samples (\a float numpy array) 1D numpy array with
        size = (number_of_samples,) normalized in [-1,1]"""
        try:
            norm_vector = audio_vector / (1.0 * 2 ** (8 * sample_width - 1))
            return norm_vector
        except Exception as e:
            print "Could not Normalize audio vector"
            raise e

    def get_norm_numpy_array_from_pydub_audio(self):
        """! Get normalized 1D numpy array corresponding to saved pydub
        audio data.
        
        \throws Exception On failure to normalize input numpy array
        \throws RuntimeError If could not convert the saved pydub audio
        data to a numpy array
        
        \returns \b norm_samples (\a float numpy array) 1D numpy array
        with size = (number_of_samples,) normalized in [-1,1]

        \warning If n_channels > 1 then the resulting numpy array 
        corresponds to channel_1_np_array | channel_2_np_array | ...
        concatenated. This should be not used externally, except if
        someone knows what he is doing. Better use: 
        get_norm_numpy_array_in_separated_channels()"""
        samples = self.get_numpy_array_from_pydub_audio()
        sample_width = self.get_sample_width()
        norm_samples = self.normalize_audio(samples, sample_width)
        return norm_samples

    def get_norm_numpy_array_in_separated_channels(self):
        """! Get a list of normalized 1D numpy arrays corresponding to
        all channels from saved pydub audio data.
        
        \throws Exception On failure to normalize input numpy array
        \throws RuntimeError If could not convert the saved pydub audio
        data to a numpy array
        
        \returns \b norm_samples_list (\a list of float numpy arrays)
        1D numpy array. Each one of them has size:
        (number_of_samples,) normalized in [-1,1]"""
        samples_list = self.get_numpy_array_in_separated_channels()
        sample_width = self.get_sample_width()
        norm_samples_list = map(lambda x:
                                self.normalize_audio(x, sample_width),
                                samples_list)
        return norm_samples_list

    @staticmethod
    def mix_channels(np_channel_list):
        """! Mix all separate channels in a 1D numpy array
        
        \param np_channel_list (\a list of 1D float numpy arrays)
        A list of numpy arrays corresponding to different channels 
        
        \throws ValueError On giving different lengths in any of the
        numpy arrays of the list
        \throws ZeroDivisionError On giving an empty list
        
        \returns \b mixed_vec (\a float numpy array) 1D numpy array with size = (number_of_samples,), all channels combined"""
        try:
            if len(np_channel_list) == 1:
                return np_channel_list[0]
            mixed_vec = sum(np_channel_list)
            mixed_vec = mixed_vec / len(np_channel_list)
            return mixed_vec
        except Exception as e:
            print "Could not Mix all the channels in one numpy array"
            raise e

    def get_mixed_channel_numpy_array(self):
        """! Get a mixed 1D numpy array corresponding to saved pydub
        audio all channels mixed in one vector.
        
        \throws RuntimeError If pydub fails to split channels
        \throws ValueError On giving different lengths in any of the
        numpy arrays of the list
        \throws ZeroDivisionError On giving an empty list
        
        \returns \b norm_samples (\a int numpy array) 1D
         numpy array with size = (number_of_samples,) with all 
         channels combined.

        \warning If n_channels > 1 then the resulting numpy array 
        corresponds to channel_1_np_array | channel_2_np_array | ...
        concatenated. This should be not used externally, except if
        someone knows what he is doing. Better use: 
        get_norm_mixed_channel_numpy_array()"""
        separated_ch_l = self.get_numpy_array_in_separated_channels()
        mixed_vec = self.mix_channels(separated_ch_l)
        return mixed_vec

    def get_norm_mixed_channel_numpy_array(self):
        """! Get a normalized mixed 1D numpy array corresponding to
         saved pydub audio all channels mixed in one vector.
        
        \throws RuntimeError If pydub fails to split channels
        \throws ValueError On giving different lengths in any of the
        numpy arrays of the list
        \throws ZeroDivisionError On giving an empty list
        \throws Exception On failure to normalize input numpy array
        
        \returns \b norm_samples (\a float numpy array) 1D
         numpy array with size = (number_of_samples,) with all 
         channels combined and normalized in [-1,1]."""
        mixed_vec = self.get_mixed_channel_numpy_array()
        sample_width = self.get_sample_width()
        norm_mixed_vec = self.normalize_audio(mixed_vec, sample_width)
        return norm_mixed_vec


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Audio Reader, reading and manipulating Audio')
    parser.add_argument("--input_path", type=str,
                        help="""Path for an audio file to be parsed""",
                        default='./sample_audio/stereo/bsi_test_audio.avi')
    parser.add_argument("--suffix", type=str,
                        help="""The suffix of your audio file. If no suffix is provided
        then it would be utomatically infered by the remaining right
        part of a character '.' split.""",
                        default=None, choices=['wav', 'mp3', 'mp4', 'ogg', 'avi', 'flac'])
    args = parser.parse_args()
    return args


def audioreader_usage_example_internal_methods(path, suffix):
    """\example Audio Reader usage example for module functionality. 
    This is not the proper way of calling from other modules. 
    You should use: audioreader_usage_example()"""
    audiofile = AudioFile(path, suffix=suffix)
    print "Parsed Audiofile: {}".format(path)

    print (("Fs = {}Hz\nNumber of Channels = {}\n"
            "Bits per Sample = {}\nDuration = {}s\n".format(
        audiofile.get_fs(), audiofile.get_n_channels(),
        audiofile.get_sample_width() * 8,
        audiofile.get_duration_seconds()
    ))
    )

    print "\n**Non normalized channel list with numpy arrays**"
    ch_l = audiofile.get_numpy_array_in_separated_channels()
    for i, ch in enumerate(ch_l):
        print ("Channel: {} \t Array Type: {}, N_Samples: {}, Max: {}"
               " Min: {}".format(i + 1, type(ch), len(ch), max(ch), min(ch)))

    print "\n**Normalized channel list with numpy arrays**"
    norm_ch_l = audiofile.get_norm_numpy_array_in_separated_channels()
    for i, ch in enumerate(norm_ch_l):
        print ("Channel: {} \t Array Type: {}, N_Samples: {}, Max: {}"
               " Min: {}".format(i + 1, type(ch), len(ch), max(ch), min(ch)))

    print "\n**Numpy array with all channels mixed**"
    ch = audiofile.get_mixed_channel_numpy_array()
    print ("Array Type: {}, N_Samples: {}, Max: {}"
           " Min: {}".format(type(ch), len(ch), max(ch), min(ch)))

    print "\n**Normalized numpy array with all channels mixed**"
    ch = audiofile.get_norm_mixed_channel_numpy_array()
    print ("Array Type: {}, N_Samples: {}, Max: {}"
           " Min: {}".format(type(ch), len(ch), max(ch), min(ch)))


def audioreader_usage_example(path, suffix):
    """\example Audio Reader usage example for module functionality 
    by using the wrapper method"""
    audiofile = AudioFile(path, suffix=suffix)
    print "Parsed Audiofile: {}".format(path)

    print (("Fs = {}Hz\nNumber of Channels = {}\n"
            "Bits per Sample = {}\nDuration = {}s\n".format(
        audiofile.get_fs(), audiofile.get_n_channels(),
        audiofile.get_sample_width() * 8,
        audiofile.get_duration_seconds()
    ))
    )

    print "\n**Non normalized channel list with numpy arrays**"
    ch_l = audiofile.read_audio_file(normalize=False,
                                     mix_channels=False)
    for i, ch in enumerate(ch_l):
        print ("Channel: {} \t Array Type: {}, N_Samples: {}, Max: {}"
               " Min: {}".format(i + 1, type(ch), len(ch), max(ch), min(ch)))

    print "\n**Normalized channel list with numpy arrays**"
    norm_ch_l = audiofile.read_audio_file(normalize=True,
                                          mix_channels=False)
    for i, ch in enumerate(norm_ch_l):
        print ("Channel: {} \t Array Type: {}, N_Samples: {}, Max: {}"
               " Min: {}".format(i + 1, type(ch), len(ch), max(ch), min(ch)))

    print "\n**Numpy array with all channels mixed**"
    ch = audiofile.read_audio_file(normalize=False, mix_channels=True)
    print ("Array Type: {}, N_Samples: {}, Max: {}"
           " Min: {}".format(type(ch), len(ch), max(ch), min(ch)))

    print "\n**Normalized numpy array with all channels mixed**"
    ch = audiofile.read_audio_file(normalize=True, mix_channels=True)
    print ("Array Type: {}, N_Samples: {}, Max: {}"
           " Min: {}".format(type(ch), len(ch), max(ch), min(ch)))


if __name__ == '__main__':
    args = get_args()
    audioreader_usage_example(args.input_path, args.suffix)
    # audioreader_usage_example_internal_methods(
    # args.input_path, args.suffix)
