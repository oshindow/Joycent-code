# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta
import librosa

from text import text_to_sequence, text_to_sequence_zh, cmudict, zhdict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram, mel_spectrogram_align
import json

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, zh_path=None):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        if zh_path is not None:
            self.zhdict = zhdict.ZHDict(zh_path)
        else:
            self.zhdict = None
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        print("text:", text)
        mel = self.get_mel(filepath)
        return (text, mel)

    def get_mel(self, filepath):
        # audio, sr = ta.load(filepath)
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        # print(audio, audio_l, sr_l)
        audio = torch.from_numpy(audio).unsqueeze(0)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=True).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        if self.zhdict is not None:
            text_norm = text_to_sequence_zh(text, dictionary=self.zhdict)
        else:
            text_norm = text_to_sequence(text, dictionary=self.cmudict)
        print("text norm:", text_norm)
        if self.add_blank:
            print(self.add_blank)
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
            print("after norm:", text_norm)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, zh_path=None):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        if zh_path is not None:
            self.zhdict = zhdict.ZHDict(zh_path)
        else:
            self.zhdict = None
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker, filepath)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram_align(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        if self.zhdict is not None:
            text_norm = text_to_sequence_zh(text, dictionary=self.zhdict)
        else:
            text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker, filepath = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker, 'filepath': filepath}
        # print(self.filelist[index])
        # print(item['y'].shape)
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk'], 
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk, 'filepath': item['filepath']}

class TextMelSpeakerAccentDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, zh_path=None, train=False):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        if zh_path is not None:
            self.zhdict = zhdict.ZHDict(zh_path)
        else:
            self.zhdict = None
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        # random.seed(random_seed)
        # random.shuffle(self.filelist)
        if train:
            self.lengths_dict = self.get_lengths()
            # self.lengths = self.write_lengths()
            self.lengths = [self.lengths_dict[key[0]] for key in self.filelist]
            self.accents = [int(key[3]) for key in self.filelist ]

    def get_triplet(self, line):
        filepath, text, speaker, accent = line[0], line[1], line[2], line[3]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        accent = self.get_accent(accent)
        return (text, mel, speaker, accent, filepath)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram_align(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel
    
    def write_lengths(self):
        self.lengths = {}
        idx = 0
        self.lengths_max = 0
        for file in self.filelist:
            if idx and idx % 1000 == 0:
                print(idx)
            mel_path = file[0]
            mel = self.get_mel(mel_path)
            length = mel.shape[1]
            self.lengths_max = max(length, self.lengths_max)
            self.lengths[mel_path] = length
            idx += 1

        print(self.lengths_max)
        with open('lengths_sg.json', 'w', encoding='utf8') as output:
            json.dump(self.lengths, output, indent=4)
            
        return self.lengths
    
    def get_lengths(self):
        with open('lengths_sg.json', 'r', encoding='utf8') as input:
            self.lengths_dict = json.load(input)
        self.lengths_max = 497
        return self.lengths_dict
    
    def get_text(self, text, add_blank=True):
        if self.zhdict is not None:
            text_norm = text_to_sequence_zh(text, dictionary=self.zhdict)
        else:
            text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            # print(text_norm)
            text_norm = intersperse(text_norm, len(self.zhdict))  # add a blank token, whose id number is len(symbols)
            # print(text_norm)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def get_accent(self, accent):
        accent = torch.LongTensor([int(accent)])
        return accent
    
    def __getitem__(self, index):
        text, mel, speaker, accent, filepath = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker, 'acc': accent, 'filepath': filepath}
        # print(self.filelist[index])
        # print(item['y'].shape)
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerAccentBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []
        acc = []
        for i, item in enumerate(batch):
            y_, x_, spk_, acc_ = item['y'], item['x'], item['spk'], item["acc"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)
            acc.append(acc_)
        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        acc = torch.cat(acc, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk, 'acc': acc, 'filepath': item['filepath']}
