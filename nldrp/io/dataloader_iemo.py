import argparse
import os
import re
import sys
from pprint import pprint

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
        self.speaker_ids = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M',
                            'Ses04F', 'Ses04M', 'Ses05F', 'Ses05M']
        self.emotion_dict = {"ang": "angry", "hap": "happy",
                             "sad": "sad", "neu": "neutral",
                             "fru": "frustrated", "exc": "excited",
                             "fea": "fearful", "sur": "surprised",
                             "dis": "disgusted", "oth": "other",
                             "xxx": "unclassified", "invalid": "invalid"}
        self.emotions_in_use = ["sad", "angry", "excited", "happy",
                                "neutral"]
        self.data_dict = self.make_data_dict()


    def get_utterance_with_lab(self,
                               annotations_dir,
                               dialogue):
        utt_labels = {}
        an_p = os.path.join(annotations_dir,
                            dialogue + '.txt')

        with open(an_p) as fd:
            an_lines = fd.readlines()

        for a_line in an_lines:
            if dialogue in a_line:
                dur, utt, em, _ = a_line.split('\t')
                utt_labels[utt] = em

        return utt_labels


    def get_info_for_utt(self,
                         utt_id,
                         true_emotion,
                         wavs_dir):

        useful = utt_id.split("_")
        fake_ses = useful[0]
        mf_number = useful[-1]
        script_name = '_'.join(useful[1:-1])

        speaker_gender = mf_number[0]
        true_session = fake_ses[:-1]
        true_speaker = true_session + speaker_gender

        fake_identifier = fake_ses + '_' + script_name

        wavpath = os.path.join(wavs_dir, fake_identifier,
                               utt_id + '.wav')
        audiofile = AudioFile(wavpath)
        fs = audiofile.get_fs()
        wav_dur = audiofile.get_duration_seconds()
        wav = audiofile.read_audio_file(normalize=True, mix_channels=True)

        info = {
            'speaker': true_speaker,
            'wavpath': wavpath,
            'emotion': true_emotion,
            'Fs': fs,
            'wav': wav,
            'wav_duration': wav_dur
        }

        return info


    def make_data_dict(self):
        data_dict = {spk: {} for spk in self.speaker_ids}
        for session in self.sessions:
            wavs_dir = os.path.join(self.data_dir, session,
                                    'sentence', 'wav')
            annotations_dir = os.path.join(self.data_dir,
                                           session, 'dialog',
                                           'EmoEvaluation')

            dialogues = os.listdir(wavs_dir)

            for dialogue in dialogues:
                utt_with_labels = self.get_utterance_with_lab(
                                  annotations_dir, dialogue)

                for utt, fake_emotion in utt_with_labels.items():
                    true_emotion = self.emotion_dict[fake_emotion]
                    if true_emotion in self.emotions_in_use:
                        try:
                            info = self.get_info_for_utt(utt,
                                                     true_emotion,
                                                     wavs_dir)
                            data_dict[info['speaker']][utt] = info

                        except Exception as e:
                            print e
                            print utt + "Failed to load"

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
    iemo_data_obj = IemocapDataLoader(iemocap_path=args.iemocap_path)
    from pprint import pprint

    dataset_dic = iemo_data_obj.data_dict
    for spkr in dataset_dic:
        print spkr, len(dataset_dic[spkr])
