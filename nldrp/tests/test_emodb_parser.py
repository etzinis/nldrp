from nldrp.io.dataloader_emodb import EmodbDataLoader

dtldr = EmodbDataLoader()

a = dtldr.data_dict
print(a['11']['a05Fb'])

emos = {'anger': 0,
        'boredom': 0,
        'disgust': 0,
        'anxiety/fear': 0,
        'happiness': 0,
        'sadness': 0,
        'neutral': 0}
for speaker in a:
    for utt in a[speaker]:
        emos[a[speaker][utt]['emotion']] += 1

print(emos)
