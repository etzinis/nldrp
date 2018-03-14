import argparse
import os
import re
import sys

nldpr_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, nldpr_dir)

from nldrp.config import BASE_PATH
from nldrp.io.AudioReader import AudioFile


class IemocapDataLoader(object):
    def __init__(self, iemocap_path=None):
        if iemocap_path is not None:
            self.data_dir = iemocap_path
        else:
            self.data_dir = os.path.join(BASE_PATH, 'dataset_iemo')
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.speaker_ids = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M', 'Ses04F', 'Ses04M', 'Ses05F',
                            'Ses05M']
        self.emotion_dict = {"ang": "angry", "hap": "happy",
                             "sad": "sad", "neu": "neutral",
                             "fru": "frustrated", "exc": "excited",
                             "fea": "fearful", "sur": "surprised",
                             "dis": "disgusted", "oth": "other",
                             "xxx": "unclassified", "invalid": "invalid"}
        self.emotions_in_use = ["sad", "angry", "excited", "happy",
                                "neutral"]
        self.data_dict = self.make_data_dict()



    def make_data_dict(self):
        data_dict = {spk: {} for spk in self.speaker_ids}
        for session in self.sessions:
            wav_dir = os.path.join(self.data_dir, session,
                                   'sentences', 'wav')
            for subsession in sorted(os.listdir(wav_dir)):
                annotation_file = os.path.join(self.data_dir, session, 'dialog', 'EmoEvaluation', subsession + '.txt')
                with open(annotation_file) as f:
                    annotations = [line for line in f if re.search(subsession, line)]
                annotations_dict = {el.split('\t')[-3]: el.split('\t')[-2] for el in annotations}
                for utt in sorted(os.listdir(os.path.join(wav_dir, subsession))):
                    if not utt.endswith('.wav'):
                        continue
                    uttid = utt.replace('.wav', '')
                    speaker = uttid.split('_', 1)[0]
                    alnum = uttid.split('_', 1)[1]
                    session = 'Session' + speaker[-2]
                    # Emotion
                    emotion = self.emotion_dict[annotations_dict[uttid]]
                    if emotion == 'unclassified' or emotion == 'other' or emotion == 'invalid':
                        continue

                    # keep only the utterances that we will use in
                    # the final classification
                    if emotion not in self.emotions_in_use:
                        continue
                    # Fs, wav, wav_duration and normalize wav to [-1,1]
                    wavpath = os.path.abspath(
                        os.path.join(self.data_dir, session,
                                     'sentences', 'wav', subsession, uttid + '.wav'))
                    audiofile = AudioFile(wavpath)
                    fs = audiofile.get_fs()
                    wav_dur = audiofile.get_duration_seconds()
                    wav = audiofile.read_audio_file(normalize=True, mix_channels=True)
                    # Dictionary construction
                    info = {
                        'speaker': speaker,
                        'wavpath': wavpath,
                        'emotion': emotion,
                        'Fs': fs,
                        'wav': wav,
                        'wav_duration': wav_dur
                    }
                    data_dict[speaker][alnum] = info
        return data_dict


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='IEMOCAP Dataset parser')
    parser.add_argument("-i", "--iemocap_path", type=str,
                        help="""The path where IEMOCAP dataset is stored""",
                        required=True)
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    """!brief Example of usage"""
    args = get_args()
    savee_data_obj = IemocapDataLoader(iemocap_path=args.iemocap_path)
    from pprint import pprint

    pprint(savee_data_obj.data_dict['Ses04M']['script01_1_F003'])
    # pprint(savee_data_obj.data_dict['DC']['a01']['phone_details'][0])
