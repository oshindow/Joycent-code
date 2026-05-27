import os
import json
accent2id = {'Changsha':1, 'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'Shanxi': 9, 'north': 10, 'south': 11, 'singapore': 12}

dataset = '/data2/xintong/aishell3/train/wav_16k'

spk2accent = json.load(open('resources/spk2accent.json', 'r', encoding='utf8'))
acc_spk_utt_dict = {}
contents = []
skip_spk = ['SSB1831', 'SSB0915', 'SSB1846', 'SSB1383',
    'SSB0817',
    'SSB1008',
    'SSB1385',
    'SSB1108',
    'SSB1630',
    'SSB0887',
'SSB0863',
'SSB1138',
'SSB0871',
'SSB1437',
'SSB1431',
'SSB1320',
'SSB1221',
'SSB1393',
'SSB1878',
'SSB1377',
'SSB1204',
'SSB1091',
'SSB1064',
'SSB0851',
'SSB1392',
'SSB1806',
'SSB1408',
'SSB1759',
'SSB1050',
'SSB1891',
'SSB1055',
'SSB1918'
]
spk2id = {}

with open('resources/filelists/zh_all/raw.txt', 'r', encoding='utf8') as input:
    for line in input:
        utt, phones, frame, mel, f0, accentid, spkid, language = line.strip().split('|')
        
        # accent = int(accentid)
        spk = utt[:7]
        # print(spk)
        if spk in skip_spk:
            continue
        
        if spk not in spk2id:
            spk2id[spk] = len(spk2id)
        
        spkid = spk2id[spk]
        acc = spk2accent[spk]
        if acc == 'others':
            continue
        if acc not in acc_spk_utt_dict:
            acc_spk_utt_dict[acc] = {}
        
        accentid = accent2id[acc]
        # if accentid not in accentid_dict:
        #     accentid_dict[accentid] = 1 
        # else:
        #     accentid_dict[accentid] += 1 
        # accent = line.strip().split('|')[0].split('_')[3]
        wavepath = os.path.join(dataset, spk, utt + '.wav')
        phonemes = phones

        content = wavepath + '|' + phonemes + '|' + str(spkid) + '|' + str(accentid)
        # contents.append(content)
        
        if spk not in acc_spk_utt_dict[acc]:
            acc_spk_utt_dict[acc][spk] = [content]
        else:
            acc_spk_utt_dict[acc][spk].append(content)



print(spk2id) # 142 个
import json
## remove others and spk with only 1 or 2 utterences. 
with open('resources/spk2id_aishell3_filtered.json', 'w', encoding='utf8') as f:    
    json.dump(spk2id, f, ensure_ascii=False, indent=4)
# train_set_unseen = []
# for accent, 
import random
test_size = 50
accent_spk_utt_cnt_unseen_test = {}
accent_spk_utt_cnt_seen_train = {}
accent_spk_utt_cnt_seen_test = {}
for accent, spk_utt in acc_spk_utt_dict.items():
    print(f"accent: {accent}, spk num: {len(spk_utt)}, utt num: {sum([len(x) for x in spk_utt.values()])}")
    test_set_seen = []
    train_set_seen = []
    test_set_unseen = []
    # find 2 unseen
    sorted_spk_utt = sorted(spk_utt.items(), key=lambda x: len(x[1]))
    # for spk, utt in sorted_spk_utt:
    #     print(spk, len(utt))
    print(sorted_spk_utt[0][0], len(sorted_spk_utt[0][1]), sorted_spk_utt[-1][0], len(sorted_spk_utt[-1][1]))

    unseen1 = sorted_spk_utt[0][0]
    unseen2 = sorted_spk_utt[1][0]
    test_set = []
    for spk, utt in spk_utt.items():
        if spk not in [unseen1, unseen2]:
            test_set_seen += random.sample(utt, test_size)
            train_set_seen += [x for x in utt if x not in test_set_seen]
        else:
            test_set_unseen += random.sample(utt, test_size)
            # train_set_unseen += [x for x in utt if x not in test_set_seen]         
    accent_spk_utt_cnt_seen_test[accent] = test_set_seen
    accent_spk_utt_cnt_seen_train[accent] = train_set_seen
    accent_spk_utt_cnt_unseen_test[accent] = test_set_unseen
    print(f"seen test num: {len(test_set_seen)}, seen train num: {len(train_set_seen)}, unseen test num: {len(test_set_unseen)}")
     

with open('resources/whisAID/aishell3/train.csv', 'w') as f:
    for acc in accent_spk_utt_cnt_seen_train:
        for line in accent_spk_utt_cnt_seen_train[acc]:
            f.write(line + '\n')
with open('resources/whisAID/aishell3/test_seen.csv', 'w') as f:
    for acc in accent_spk_utt_cnt_seen_test:
        for line in accent_spk_utt_cnt_seen_test[acc]:
            f.write(line + '\n')
with open('resources/whisAID/aishell3/test_unseen.csv', 'w') as f:
    for acc in accent_spk_utt_cnt_unseen_test:
        for line in accent_spk_utt_cnt_unseen_test[acc]:
            f.write(line + '\n')
            
