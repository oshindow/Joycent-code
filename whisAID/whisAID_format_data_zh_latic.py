import os
import json
accent2id = {'Changsha':1, 'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'Shanxi': 9, 'north': 10, 'south': 11, 'singapore': 12, 'nonnative': 13}

import os

dataset = '/path/to/data/LATIC/WAVE/WAVE'
# waves = {'G0001':[], 'G0002':[], 'G0003':[], 'G0004':[]}
spk2id = {'0010':292, '0011':293, '0012':294, '0013':295}
accentid = 13

spk_utt_dict = {}

with open('/path/to/data/accented_mdd/dump2/final_labels/latic', 'r', encoding='utf8') as input:
    for line in input:
        uid, text = line.strip().split('\t') 
        spk = uid[1:5]
        wavepath = "/path/to/data/LATIC/WAVE/WAVE/SPEAKER" + spk + '/SESSION0/' + uid + '.WAV' 
                
        spkid = spk2id[spk]
        # wavepath = os.path.join(dataset, spk, line.strip().split('|')[0] + '.wav')
        phonemes = text

        content = wavepath + '|' + phonemes + '|' + str(spkid) + '|' + str(accentid)
        if spk not in spk_utt_dict:
            spk_utt_dict[spk] = [content]
        else:
            spk_utt_dict[spk].append(content)

   
import random
test_size = 50
  

print(f"utt num: {sum([len(x) for x in spk_utt_dict.values()])}")
test_set_seen = []
train_set_seen = []
test_set_unseen = []
# find 2 unseen
sorted_spk_utt = sorted(spk_utt_dict.items(), key=lambda x: len(x[1]))
# for spk, utt in sorted_spk_utt:
#     print(spk, len(utt))
print(sorted_spk_utt[0][0], len(sorted_spk_utt[0][1]), sorted_spk_utt[-1][0], len(sorted_spk_utt[-1][1]))

unseen1 = sorted_spk_utt[0][0]
unseen2 = sorted_spk_utt[1][0]
test_set = []
for spk, utt in spk_utt_dict.items():
    if spk not in [unseen1, unseen2]:
        test_set_seen += random.sample(utt, test_size)
        train_set_seen += [x for x in utt if x not in test_set_seen]
    else:
        test_set_unseen += random.sample(utt, test_size)
        # train_set_unseen += [x for x in utt if x not in test_set_seen]         
accent_spk_utt_cnt_seen_test  = test_set_seen
accent_spk_utt_cnt_seen_train  = train_set_seen
accent_spk_utt_cnt_unseen_test  = test_set_unseen
print(f"seen test num: {len(test_set_seen)}, seen train num: {len(train_set_seen)}, unseen test num: {len(test_set_unseen)}")
     

with open('resources/whisAID/latic/train.csv', 'w') as f:
    # for acc in accent_spk_utt_cnt_seen_train:
        for line in accent_spk_utt_cnt_seen_train:
            f.write(line + '\n')
with open('resources/whisAID/latic/test_seen.csv', 'w') as f:
    # for acc in accent_spk_utt_cnt_seen_test:
        for line in accent_spk_utt_cnt_seen_test:
            f.write(line + '\n')
with open('resources/whisAID/latic/test_unseen.csv', 'w') as f:
    # for acc in accent_spk_utt_cnt_unseen_test:
        for line in accent_spk_utt_cnt_unseen_test:
            f.write(line + '\n')
            
