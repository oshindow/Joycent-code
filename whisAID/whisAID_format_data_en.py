import os
import json

commonaccent = '/path/to/data/commonvoice17/'
uttid2path = {}
idx = 0
for root, dirs, files in os.walk(commonaccent):
    for file in files:
        # print(file)
        
        if 'mp3' in file and 'en' in file:
            # if idx % 1000 == 0:
            #     print(idx, len(uttid2path))
            uttid2path[file] = os.path.join(root, file)
            idx += 1

print(len(uttid2path))



accent2id = {}
spk2id = {}
# output = open('resources/whisAID/CommonAccent/train.csv', 'w', encoding='utf8')
with open('resources/CommonAccent/train.csv', 'r', encoding='utf8') as input:
    for line in input:
        try:
            ID,utt_id,wav,wav_format,text,duration,speaker,gender,accent = line.strip().split(',')
        except Exception as e:
            continue
        if speaker not in spk2id:
            spk2id[speaker] = len(spk2id)
        # else:
        spkid = spk2id[speaker]

        if accent not in accent2id:
            accent2id[accent] = len(accent2id)
        
        accid = accent2id[accent]
        if wav.split('/')[-1] in uttid2path:
            path = uttid2path[wav.split('/')[-1]]
        else: 
            print(wav)
            continue
        content = path + '|' + text + '|' + str(spkid) + '|' + str(accid) 
        # output.write(content + '\n')

# output.close()
print(accent2id, len(spk2id))
