import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
from collections import defaultdict
def get_audio_lengths(file_path):
    audio = AudioSegment.from_file(file_path)
    length_in_seconds = len(audio) / 1000  # Length in seconds
    # audio_lengths.append(length_in_seconds)

    return length_in_seconds

# 初始化字典
spk_dict = {}
acc_dict = {}
length_dict = {}
acc_spk = {'0': set(),'1': set(),'2':set() ,'3': set()}

# 读取文件并统计
root = '/data2/xintong/magichub_singapore/wav_16k/G0001'
import os
for file in os.listdir(root):
    if '.wav' in file:
        filepath = os.path.join(root, file)
        # uid, text, acc = line.strip().split('|')
        spk = filepath.split('/')[-2]
        spk_dict[spk] = spk_dict.get(spk, 0) + 1
        # acc_dict[acc] = acc_dict.get(acc, 0) + 1
        length_dict[file] = get_audio_lengths(filepath)
        # acc_spk[acc].add(spk)
import json
# 打印统计结果
print("acc_dict:", acc_dict)
print("spk_dict:", spk_dict)
# print("length_dict:", length_dict)
print('length_dict', sum(length_dict.values())/ 60)
# with open('G0001_lengths.json', 'w', encoding='utf8') as output:
#     json.dump(length_dict, output, indent=2)

## extract test files
import shutil
def extract_test_speech(root, file_path):
    # data = {}
    test_data = json.load(open(file_path, 'r', encoding='utf8'))
    for key, value, in test_data.items():
        old_path = os.path.join(root, key)
        new_path = old_path.replace('wav_16k', 'G0004_test')
        shutil.copy(old_path, new_path)

root = '/data2/xintong/magichub_singapore/wav_16k/G0004'
file_path = '/data2/xintong/magichub_singapore/G0004_part2_test.json'
extract_test_speech(root, file_path)

## alignment2textgrid
import tgt
import re
def alignment2textgrid(root, file_path):
    """
    root: alignment files root
    file_path: test json file path
    """
    with open(file_path, 'r', encoding='utf8') as input:
        test_data = json.load(input).keys()
    
    
    for file in os.listdir(root):
        if file[:-4] + '.wav' in test_data:
            phoneme_alignments = []
            char_alignments = []

            alignmentfilepath = os.path.join(root, file)
            charfilepath = alignmentfilepath.replace('alignments', 'texts')
            start_in_sec = 0
            with open(alignmentfilepath, 'r', encoding='utf8') as input:
                for line in input:
                    phoneme, duration = line.strip().split(' ')
                    end_in_sec = start_in_sec + float(duration)
                    phoneme_alignments.append((phoneme, start_in_sec, end_in_sec))
                    start_in_sec = end_in_sec

            with open(charfilepath, 'r', encoding='utf8') as input:
                data = input.readline().strip()
                cleaned_data = re.sub(r'[^\u4e00-\u9fff]', '', data)
                chars = [char for char in cleaned_data]
            
            # align char with phone
            idx_phoneme = 0
            idx_char = 0
            while idx_phoneme < len(phoneme_alignments) and idx_char <= len(chars):
            # for idx in range(len(phoneme_alignments)):
                if phoneme_alignments[idx_phoneme][0] == 'sil':
                    char_alignments.append(phoneme_alignments[idx_phoneme])
                    idx_phoneme += 1
                else:
                    char_alignments.append((chars[idx_char], phoneme_alignments[idx_phoneme][1], phoneme_alignments[idx_phoneme + 1][2]))
                    idx_phoneme += 2
                    idx_char += 1
            
            # tgt file
            tgtfilepath = alignmentfilepath.replace('alignments', 'textgrids')
            if not os.path.isdir(os.path.dirname(tgtfilepath)):
                os.makedirs(os.path.dirname(tgtfilepath))
            textgrid = tgt.core.TextGrid(filename=tgtfilepath)
            # phoneme tier
            intervaltier = tgt.core.IntervalTier(start_time=float(phoneme_alignments[0][1]), end_time=phoneme_alignments[-1][2], name='phoneme')
            for line in phoneme_alignments:
                phone, start, end = line
                interval = tgt.core.Interval(start_time=float(start), end_time=float(end), text=phone)
                try:
                    intervaltier.add_interval(interval)
                except Exception as e:
                    print(interval, line)
            textgrid.add_tier(intervaltier)
            # white tier
            intervaltier = tgt.core.IntervalTier(start_time=float(phoneme_alignments[0][1]), end_time=phoneme_alignments[-1][2], name='man_phoneme')
            for line in phoneme_alignments:
                phone, start, end = line
                interval = tgt.core.Interval(start_time=float(start), end_time=float(end), text=phone)
                try:
                    intervaltier.add_interval(interval)
                except Exception as e:
                    print(interval, line)
            textgrid.add_tier(intervaltier)
            # char tier
            intervaltier = tgt.core.IntervalTier(start_time=float(char_alignments[0][1]), end_time=char_alignments[-1][2], name='char')
            for line in char_alignments:
                char, start, end = line
                interval = tgt.core.Interval(start_time=float(start), end_time=float(end), text=char)
                try:
                    intervaltier.add_interval(interval)
                except Exception as e:
                    print(interval, line)
            textgrid.add_tier(intervaltier)
            
            tgt.io.write_to_file(textgrid, tgtfilepath, format='long', encoding='utf-8')

# Read a TextGrid object from a file.
root = '/data2/xintong/magichub_singapore/alignments/G0004'
file_path = '/data2/xintong/magichub_singapore/G0004_part2_test.json'
alignment2textgrid(root, file_path)
print('')
## split contents
def split_contents(content_path):
    with open(content_path, 'r', encoding='utf8') as input:
        for line in input:
            wavefile, text = line.strip().split('\t')
            textfilepath = wavefile.replace('WAV_segments', 'texts')
            textfilepath = textfilepath.replace('wav', 'txt')
            # print(textfilepath)
            if not os.path.isdir(os.path.dirname(textfilepath)):
                print('making dir', os.path.dirname(textfilepath))
                os.makedirs(os.path.dirname(textfilepath))
            with open(textfilepath, 'w', encoding='utf8') as output:
                output.write(text)

content_path = '/data2/xintong/magichub_singapore/content.txt'
split_contents(content_path) 
# import random
# length_list = list(length_dict.items())
# random.shuffle(length_list)
# def split_samples_to_sum_40(samples):
#     part1 = []
#     part2 = []
#     total_sum = 0

#     for value in samples:
#         if total_sum + value[1] <= 40 * 60:
#             part1.append(value)
#             total_sum += value[1]
#         else:
#             part2.append(value)
    
#     return part1, part2

# Generate random samples
# samples = length_dict
# part1, part2 = split_samples_to_sum_40(length_list)

# # print("Original Samples:", samples)
# part2 = dict(part2)
# part1 = dict(part1)
# with open('G0004_part1_label.json', 'w', encoding='utf8') as output:
#     json.dump(part1, output, indent=2)
# with open('G0004_part2_test.json', 'w', encoding='utf8') as output:
#     json.dump(part2, output, indent=2)    
# print("Part 1 (sum <= 40):", len(part1), sum(part1.values())/60/60)
# print("Part 2 (remaining):", len(part2), sum(part2.values())/60/60)
# print(sum(part1.values())/60 + sum(part2.values())/60, sum(length_dict.values())/ 60)
# 
# G0004 2.269094547964116
# G0002 2.1751876056338038
# G0003 1.9881709136109338
# G0001 1.6910552036199087
# print("length_dict:", [len(spk) for spk in acc_spk.values()])

# 画柱状图
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# # 绘制 acc_dict 的柱状图
# ax1.bar(acc_dict.keys(), acc_dict.values())
# ax1.set_title('Distribution of Accents')
# ax1.set_xlabel('Accent')
# ax1.set_ylabel('Count')
# ax1.set_xticklabels(acc_dict.keys(), rotation=45)

# # 绘制 spk_dict 的柱状图
# ax2.bar(spk_dict.keys(), spk_dict.values())
# ax2.set_title('Distribution of Speakers')
# ax2.set_xlabel('Speaker')
# ax2.set_ylabel('Count')
# ax2.set_xticklabels(spk_dict.keys(), rotation=45)

# # 调整布局
# plt.tight_layout()
# plt.savefig('data_distribution_dedup.png')
