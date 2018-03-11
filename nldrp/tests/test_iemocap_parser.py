from nldrp.io.dataloader_iemo import IemocapDataLoader

dtldr = IemocapDataLoader()

a = dtldr.data_dict
spkr = 'Ses04M'
uttid = 'script01_1_F003'
print(a['Ses03M']['impro05b_M015']['emotion'])

emos = {"angry": 0,
        "happy": 0,
        "sad": 0,
        "neutral": 0,
        "frustrated": 0,
        "excited": 0,
        "fearful": 0,
        "surprised": 0,
        "disgusted": 0}
for speaker in a:
    for utt in a[speaker]:
        emos[a[speaker][utt]['emotion']] += 1

print(emos)
