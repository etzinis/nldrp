import argparse
import os
import sys

nldpr_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, nldpr_dir)

from nldrp.config import BASE_PATH
from nldrp.io.AudioReader import AudioFile


class EmodbDataLoader(object):
    def __init__(self, emodb_path=None):
        if emodb_path is not None:
            self.data_dir = emodb_path
        else:
            self.data_dir = os.path.join(BASE_PATH, 'dataset_emodb')
        self.speaker_ids = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
        self.emotion_dict = {'W': 'anger',
                             'L': 'boredom',
                             'E': 'disgust',
                             'A': 'anxiety/fear',
                             'F': 'happiness',
                             'T': 'sadness',
                             'N': 'neutral'}
        self.data_dict = self.make_data_dict()

    def make_data_dict(self):
        data_dict = {spk: {} for spk in self.speaker_ids}
        wav_list = sorted(os.listdir(os.path.join(self.data_dir, 'wav')))
        for f in wav_list:
            uttid = f.replace('.wav', '')
            speaker = str(uttid[0:2])
            alnum = str(uttid[2:])
            transcr = str(uttid[2:5])
            emotion = self.emotion_dict[str(uttid[5])]
            version = str(uttid[6])
            # Fs, wav, wav_duration and normalize wav to [-1,1]
            wavpath = os.path.abspath(os.path.join(self.data_dir, 'wav', uttid + '.wav'))
            audiofile = AudioFile(wavpath)
            fs = audiofile.get_fs()
            wav_dur = audiofile.get_duration_seconds()
            wav = audiofile.read_audio_file(normalize=True, mix_channels=True)
            info = {
                'speaker': speaker,
                'wavpath': wavpath,
                'emotion': emotion,
                'Fs': fs,
                'wav': wav,
                'wav_duration': wav_dur,
                'transcr': transcr,
                'version': version
            }
            data_dict[speaker][alnum] = info

        return data_dict


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='EMODB Dataset parser')
    parser.add_argument("-i", "--emodb_path", type=str,
                        help="""The path where EMODB dataset is stored""",
                        required=True)
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    savee_data_obj = EmodbDataLoader(emodb_path=args.emodb_path)
    from pprint import pprint

    pprint(savee_data_obj.data_dict['11']['a05Fb'])
    # pprint(savee_data_obj.data_dict['DC']['a01']['phone_details'][0])
