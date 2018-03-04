import argparse
import numpy as np
import os
import sys

nldpr_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, nldpr_dir)

from nldrp.config import BASE_PATH
from nldrp.io.AudioReader import AudioFile


class SaveeDataloader(object):
    def __init__(self, savee_path=None):
        if savee_path is not None:
            self.data_dir = savee_path
        else:
            self.data_dir = os.path.join(BASE_PATH, 'dataset')
        self.speaker_ids = ['DC', 'JE', 'JK', 'KL']
        self.emotion_ids = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
        self.emotion_dict = {
            'a': 'anger',
            'd': 'disgust',
            'f': 'fear',
            'h': 'happiness',
            'n': 'neutral',
            'sa': 'sadness',
            'su': 'surprise'
        }
        self.uttids = self.make_uttids()
        self.data_dict = self.make_data_dict()

    def make_data_dict(self):
        data_dict = {spk: {} for spk in self.speaker_ids}
        for uttid in self.uttids:
            speaker = uttid.split('_')[0]
            alnum = uttid.split('_')[1]
            # Fs, wav, wav_duration and normalize wav to [-1,1]
            wavpath = os.path.join(self.data_dir, 'AudioData',
                                   speaker, alnum + '.wav')
            audiofile = AudioFile(wavpath)
            fs = audiofile.get_fs()
            wav_dur = audiofile.get_duration_seconds()
            wav = audiofile.read_audio_file(normalize=True, mix_channels=True)
            # F0 track
            f0path = os.path.join(self.data_dir, 'Annotation',
                                  speaker, 'FrequencyTrack',
                                  alnum.replace('0', '') + '.txt')
            # Phones
            phonepath = os.path.join(self.data_dir, 'Annotation',
                                     speaker, 'PhoneticLabel',
                                     alnum.replace('0', '') + '.txt')
            phone_specific = self.get_phone_specific(phonepath, wav, fs, wav_dur)
            # Dictionary construction
            info = {
                'speaker': speaker,
                'wavpath': wavpath,
                'emotion': self.emotion_dict[alnum[0:-2]],
                'Fs': fs,
                'wav': wav,
                'wav_duration': wav_dur,
                'F0': np.loadtxt(f0path, dtype='float64'),
                'phone_start_times': np.array(phone_specific['phone_start_times']),
                'phone_labels': phone_specific['phone_labels'],
                'phone_details': phone_specific['phone_details']
            }
            data_dict[speaker][alnum] = info
        return data_dict

    def make_uttids(self):
        uttids = ['{0}_{1}{2:02d}'.format(speaker, emotion, num)
                  for speaker in self.speaker_ids
                  for emotion in self.emotion_ids
                  for num in range(1, 16)]
        uttids.extend(['{0}_n{1:02d}'.format(speaker, num)
                       for speaker in self.speaker_ids
                       for num in range(16, 31)])
        uttids.sort()
        return uttids

    @staticmethod
    def get_phone_specific(phonepath, wav, fs, wav_dur):
        # Read phone order
        with open(phonepath) as f:
            lines = f.readlines()
        phone_annotation = [line.strip() for line in lines]
        phone_labels = [line.split(' ')[1] for line in phone_annotation]
        # Read phone starting times
        phone_start_times = [float(line.split(' ')[0]) for line in phone_annotation]
        # Make phone list of dictionaries
        times = list(phone_start_times)
        times.append(wav_dur)
        phone_details = []
        for i, phone in enumerate(phone_labels):
            start = times[i]
            end = times[i + 1]
            start_sample = int(round(start * fs))
            end_sample = int(round(end * fs))
            phone_details.append({
                'phone_label': phone,
                'start': start,
                'start_sample': start_sample,
                'end': end,
                'end_sample': end_sample,
                'nparray': np.array(wav[start_sample:end_sample])
            })
        phone_specific = {
            'phone_labels': phone_labels,
            'phone_start_times': phone_start_times,
            'phone_details': phone_details
        }
        return phone_specific


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='SAVEE Dataset parser')
    parser.add_argument("-i", "--savee_path", type=str,
                        help="""The path where SAVEE dataset is stored""",
                        required=True)
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    savee_data_obj = SaveeDataloader(savee_path=args.savee_path)
    from pprint import pprint

    pprint(savee_data_obj.data_dict['KL']['a05'])
    # pprint(savee_data_obj.data_dict['DC']['a01']['phone_details'][0])
