import os
import numpy as np
import scipy.io.wavfile

from nldrp.config import BASE_PATH


class SaveeDataloader(object):
    def __init__(self):
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
        data_dict = {}
        for uttid in self.uttids:
            speaker = uttid.split('_')[0]
            alnum = uttid.split('_')[1]
            wavpath = os.path.join(self.data_dir, 'AudioData',
                                   speaker, alnum + '.wav')
            fs, wav = scipy.io.wavfile.read(wavpath)
            f0path = os.path.join(self.data_dir, 'Annotation',
                                  speaker, 'FrequencyTrack',
                                  alnum.replace('0', '') + '.txt')
            phonepath = os.path.join(self.data_dir, 'Annotation',
                                     speaker, 'PhoneticLabel',
                                     alnum.replace('0', '') + '.txt')
            with open(phonepath) as f:
                lines = f.readlines()
            phone_annotation = [line.strip() for line in lines]
            info = {
                'speaker': speaker,
                'wavpath': wavpath,
                'emotion': alnum[0:-2],
                'Fs': fs,
                'wav': wav,
                'F0': np.loadtxt(f0path, dtype='float64'),
                'phone_time': np.array([float(line.split(' ')[0])
                                        for line in phone_annotation]),
                'phone_annotation': [line.split(' ')[1]
                                     for line in phone_annotation]
            }
            data_dict[uttid] = info
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
