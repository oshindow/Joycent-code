data1 = ['/path/to/data/Changsha_Dialect_Conversational_Speech_Corpus',
         '/path/to/data/Guangzhou_Cantonese_Conversational_Speech_Corpus',
         '/path/to/data/Nanchang_Dialect_Conversational_Speech_Corpus',
         '/path/to/data/Shanghai_Dialect_Conversational_Speech_Corpus',
         '/path/to/data/Sichuan_Dialect_Conversational_Speech_Corpus',
         '/path/to/data/Tianjin_Dialect_Conversational_Speech_Corpus',
         '/path/to/data/Zhengzhou_Dialect_Conversational_Speech_Corpus',]

data2 = ['/path/to/data/Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence',
         '/path/to/data/Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle',
         '/path/to/data/Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence',
         '/path/to/data/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence',
         '/path/to/data/Tianjin_Dialect_Speech_Corpus_for_TTS',
         '/path/to/data/Wuhan_Dialect_Scripted_Speech_Corpus',
         '/path/to/data/Zhengzhou_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence']

acc2id = {'Changsha':1, 'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'Shanxi': 9}

import os

datas = []
spk_idx = 142
spk2id = {}
accent_spk_utt_cnt = {}
for dataset in data1:
    acc = dataset.split('_')[0].split('/')[-1]
    if acc == 'Guangzhou':
        acc = 'Guangdong'
    elif acc == 'Zhengzhou':
        acc = 'Henan'
    accid = acc2id[acc]
    if acc not in accent_spk_utt_cnt:
        accent_spk_utt_cnt[acc] = {}
    
    for file in os.listdir(os.path.join(dataset, 'wav_16k')):
        if not file.endswith('.wav'):
            continue
        spk = dataset.split('/')[-1] + '_' + file.split('_')[3]
        if spk not in spk2id:
            spk2id[spk] = spk_idx
            spk_idx += 1
        if spk not in accent_spk_utt_cnt[acc]:
            accent_spk_utt_cnt[acc][spk] = ["|".join([os.path.join(dataset, 'wav_16k', file), str(spk2id[spk]), str(accid)])]
        else:
            accent_spk_utt_cnt[acc][spk].append("|".join([os.path.join(dataset, 'wav_16k', file), str(spk2id[spk]), str(accid)]))
        # datas.append("|".join([os.path.join(dataset, 'wav_16k', file), 'sil', str(spk2id[spk]), str(accid)]))
    

for dataset in data2:
    acc = dataset.split('_')[0].split('/')[-1]
    if acc == 'Guangzhou':
        acc = 'Guangdong'
    elif acc == 'Zhengzhou':
        acc = 'Henan'
    if acc not in accent_spk_utt_cnt:
        accent_spk_utt_cnt[acc] = {}
    accid = acc2id[acc]
    # spk2id = {}
    spk_utt_cnt = {}
    if 'Tianjin' in dataset:
        for root, dir, files in os.walk(os.path.join(dataset, 'WAV')):
            for file in files:
                # print(dir, files)
                spk = dataset.split('/')[-1] 
                if spk not in spk2id:
                    spk2id[spk] = spk_idx
                    spk_idx += 1
                if spk not in accent_spk_utt_cnt[acc]:
                    accent_spk_utt_cnt[acc][spk] = ["|".join([os.path.join(dataset, 'WAV', file), str(spk2id[spk]), str(accid)])]
                else:
                    accent_spk_utt_cnt[acc][spk].append("|".join([os.path.join(dataset, 'WAV', file), str(spk2id[spk]), str(accid)]))
                # datas.append("|".join([os.path.join(dataset, 'WAV', file),'sil', str(spk2id[spk]), str(accid)]))

    else:
        for root, dir, files in os.walk(os.path.join(dataset, 'WAV')):
            for file in files:
                # print(dir, files)
                spk = dataset.split('/')[-1] + '_' + file.split('_')[0]
                if spk not in spk2id:
                    spk2id[spk] = spk_idx
                    spk_idx += 1
                if spk not in accent_spk_utt_cnt[acc]:
                    accent_spk_utt_cnt[acc][spk] = ["|".join([os.path.join(dataset, 'WAV', file.split('_')[0], file), str(spk2id[spk]), str(accid)])]
                else:
                    accent_spk_utt_cnt[acc][spk].append("|".join([os.path.join(dataset, 'WAV', file.split('_')[0], file), str(spk2id[spk]), str(accid)]))
                # datas.append("|".join([os.path.join(dataset, 'WAV', file.split('_')[0], file),'sil', str(spk2id[spk]), str(accid)]))
     
spk2acc = {}
spk2acc_name = {}
with open('/path/to/data/Mandarin_Heavy_Accent_Conversational_Speech_Corpus/SPKINFO.txt', 'r') as f:
    for line in f:
        channel, spk, gen, age, place, res, device = line.strip().split('\t')
        if not place.startswith('CHINA'):
            continue
        acc = place.split(',')[1]
        accid = acc2id[acc]
        spk = 'Mandarin_Heavy_Accent_' + spk
         
        if spk not in spk2acc:
            spk2acc[spk] = accid   
            spk2acc_name[spk] = acc
heavy = '/path/to/data/Mandarin_Heavy_Accent_Conversational_Speech_Corpus'
# spk_utt_cnt = {}
for file in os.listdir(os.path.join(heavy, 'wav_16k')):
    if not file.endswith('.wav'):
        continue
    spk = 'Mandarin_Heavy_Accent_' + file.split('_')[3]
    if spk not in spk2id:
        spk2id[spk] = spk_idx
        spk_idx += 1
    
    
    accid = spk2acc[spk]
    acc_name = spk2acc_name[spk]
    
    if acc_name not in accent_spk_utt_cnt:
        accent_spk_utt_cnt[acc_name] = {}
    
    if spk not in accent_spk_utt_cnt[acc_name]:
        accent_spk_utt_cnt[acc_name][spk] = ["|".join([os.path.join(heavy, 'wav_16k', file), str(spk2id[spk]), str(accid)])]
    else:
        accent_spk_utt_cnt[acc_name][spk].append("|".join([os.path.join(heavy, 'wav_16k', file), str(spk2id[spk]), str(accid)]))
    # datas.append("|".join([os.path.join(heavy, 'wav_16k', file), 'sil', str(spk2id[spk]), str(accid)]))
import librosa
# print("processing...", dataset)
acc_item = 0
duration_total = 0
for acc in accent_spk_utt_cnt:
    print("acc:", acc, "spk num:", len(accent_spk_utt_cnt[acc])) 
    items = 0
    duration_all = 0
    for spk in accent_spk_utt_cnt[acc]:
        items += len(accent_spk_utt_cnt[acc][spk])
        for utt in accent_spk_utt_cnt[acc][spk]:
            wavepath = utt.split('|')[0]
            y, sr = librosa.load(wavepath, sr=None)
            # if sr != 16000:
            #     print("sr != 16000", acc)
            duration = len(y) / sr
            duration_all += duration
    # print()
    print("total utts:", items)
    print("total duration:", duration_all / 3600)
    duration_total += duration_all
    acc_item += items
print(duration_total)
print(acc_item)
# select speakers (seen/unseen) 
import random

accent_spk_utt_cnt_unseen_test = {}
accent_spk_utt_cnt_seen_train = {}
accent_spk_utt_cnt_seen_test = {}
test_size = 50
for accent, spk_utt in accent_spk_utt_cnt.items():
    if len(spk_utt) < 7:
        print(f"skip accent: {accent} for unseen, spk num: {len(spk_utt)}, utt num: {sum([len(x) for x in spk_utt.values()])}")
         
        test_set = []
        train_set = []
        for spk, utt in spk_utt.items():
            test_set += random.sample(utt, test_size)
            train_set += [x for x in utt if x not in test_set]
        
        accent_spk_utt_cnt_seen_test[accent] = test_set
        accent_spk_utt_cnt_seen_train[accent] = train_set
        print(f"seen test num: {len(test_set)}, seen train num: {len(train_set)}")
        continue
    
    print(f"accent: {accent}, spk num: {len(spk_utt)}, utt num: {sum([len(x) for x in spk_utt.values()])}")
    # find 2 unseen
    sorted_spk_utt = sorted(spk_utt.items(), key=lambda x: len(x[1]))
    print(sorted_spk_utt[0][0], len(sorted_spk_utt[0][1]), sorted_spk_utt[-1][0], len(sorted_spk_utt[-1][1]))

    unseen1 = sorted_spk_utt[0][0]
    unseen2 = sorted_spk_utt[1][0]

    # seen_set
     
    test_set_seen = []
    train_set_seen = []
    test_set_unseen = []
    # train_set_unseen = []
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


with open('resources/whisAID/accent_zh/train.csv', 'w') as f:
    for acc in accent_spk_utt_cnt_seen_train:
        for line in accent_spk_utt_cnt_seen_train[acc]:
            f.write(line + '\n')
with open('resources/whisAID/accent_zh/test_seen.csv', 'w') as f:
    for acc in accent_spk_utt_cnt_seen_test:
        for line in accent_spk_utt_cnt_seen_test[acc]:
            f.write(line + '\n')
with open('resources/whisAID/accent_zh/test_unseen.csv', 'w') as f:
    for acc in accent_spk_utt_cnt_unseen_test:
        for line in accent_spk_utt_cnt_unseen_test[acc]:
            f.write(line + '\n')
            
# with open('resources/filelists/zh_all/train.accents.rmaishell3', 'w') as f:
#     for line in train_set:
#         f.write(line + '\n')

# with open('resources/filelists/zh_all/valid.accents.rmaishell3', 'w') as f:
#     for line in test_set:
#         f.write(line + '\n')

# import json
# # with open('resources/filelists/spk2acc.heavy.json', 'w') as f:
# #     json.dump(spk2acc, f, indent=4)
# with open('resources/spk2id_acc.json', 'w') as f:
#     json.dump(spk2id, f, indent=4)
