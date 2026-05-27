
import random
import torch
import torchaudio as ta
import whisper
import numpy as np
import os


def get_data_lists(text_paths, task='train'):

    samples = []
    if task == 'train':
        # print("train text paths:", text_paths)
        with open(text_paths, 'r', encoding='utf-8-sig') as input:
            for line in input:
                if 'asr_chinese' in line:
                    line = line.replace('train', 'train/')
                samples.append(line.strip())
        return samples

    for text_path in text_paths:
        # print("text paths:", text_paths)
        with open(text_path, 'r', encoding='utf-8-sig') as input:
            # print(text_path)
            for line in input:
                samples.append(line.strip())
                # uid = line.strip().split()[0].strip()
                # data = line.strip().split()[1:]

                # texts = [item for item in data]
                
                # if 'aishell3' in text_path:
                #     spk = uid[:7]
                #     filepath = "/data2/xintong/tts_chinese/aishell3/" + task + '/wav_16k/' + spk + '/' + uid + '.wav'
                
                # if 'magichub' in text_path:
                    # spk = uid.split('_')[3]
                    # filepath = "/data2/xintong/magichub_singapore/clean_data/clean_data/wav_16k/" + spk + '/' + uid  
                
                # elif 'latic' in text_path:
                #     spk = uid[1:5]
                #     filepath = "/data2/xintong/LATIC/WAVE/WAVE/SPEAKER" + spk + '/SESSION0/' + uid + '.WAV' 
                
                # elif 'sichuan' in text_path:
                #     spk = uid.split('_')[0]
                #     filepath = "/data1/xintong/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence/WAV/" + spk + '/' + uid  
                
                # elif 'heavy' in text_path:
                #     # print(text_path)
                #     # spk = uid.split('_')[0]
                #     filepath = "/data2/xintong/Mandarin_Heavy_Accent_Conversational_Speech_Corpus/wav_16k/" + uid  
                # elif "aishell1" in text_path:
                    # filepath = uid
                    # print(filepath)
                # samples.append("|".join([filepath, " ".join(texts)]))

    # with open(task + '_data', 'w', encoding="utf8") as output:
    #     for sample in samples:
    #         output.write(sample + '\n')
    
    return samples


class WhisperPinyinDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_paths, tokenizer, spk_info_path, config, task='train', pseudo_labels=None, random_seed=1020):

        self.datalist = get_data_lists(filelist_paths, task=task)
         
        self.config = config
        random.seed(random_seed)
        random.shuffle(self.datalist)
        self.accent_count = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    def __getitem__(self, index):
        """
        return:
            input_ids: Tensor (Dim, T) by default dim=80
            dec_input_ids: list, [50260, 50359, 50363, 9572, 220, 24726, 220, 18681, 220, 14028, 26923, 220, 12579, 22933, 220, 6404, 19021, 220, 15686] <|startoftranscript|><|zh|><|transcribe|><|notimestamps|>tokens
            labels: list, [50258, 50260, 50359, 50363, 9572, 220, 24726, 220, 18681, 220, 14028, 26923, 220, 12579, 22933, 220, 6404, 19021, 220, 15686, 50257] <|zh|><|transcribe|><|notimestamps|>tokens<|endoftext|>
        """
        # print(self.datalist[index])
        # print(self.datalist[index])
        audiofile, texts, spk, acc = self.datalist[index].split("|")
        
        uid = audiofile.split('/')[-1][:-4]
        if '.mp3' in audiofile:
            audiofile = audiofile.replace('.mp3', '.wav')
        accent = int(acc)
        spk = int(spk)
        
        # print(audiofile)
        if not os.path.isfile(audiofile):
            part = audiofile.split('/')[4]  # 5 if on gpu2, 4 if on gpu7
            # print(part)
            audiofile1 = audiofile.replace(part, 'dev')
            audiofile2 = audiofile.replace(part, 'train')
            audiofile3 = audiofile.replace(part, 'test')
            if os.path.isfile(audiofile1):
                # print("is", audiofile1)
                audiofile = audiofile1
            elif os.path.isfile(audiofile2):
                # print("is", audiofile2)
                audiofile = audiofile2
            elif os.path.isfile(audiofile3):
                # print("is", audiofile3)
                audiofile = audiofile3
             
        audio, sr = ta.load(audiofile)
        
        # assert audio.shape[0] == 1 # mono channel
        if audio.shape[0] > 1:
            # print("not mono channel", audiofile)
            audio = audio.mean(dim=0, keepdim=True)

        # 如果不是16k就重采样
        if sr != 16000:
            # print("not 16k sampling rate", audiofile)
            resampler = ta.transforms.Resample(sr, 16000)
            audio = resampler(audio)

        # print(audiofile, audio.shape, audio.min(), audio.max(), sr)
        # assert sr == 16000
        duration = audio.shape[-1] / 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.config.n_mels) # torch.Size([128, 3000])
   
        return {
            "audiofile": audiofile,
            "durations": duration,
            "uids": audiofile,
            "input_ids": mel,
            "accent_labels": accent,
            "spk_ids": spk
        }

    def __len__(self):
        return len(self.datalist)

class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        durations, uids, input_ids, accent_labels, spks, audiofiles = [], [], [], [], [], [] 
        for f in features:
            audiofiles.append(f["audiofile"])
            durations.append(f["durations"])
            uids.append(f["uids"]) 
            input_ids.append(f["input_ids"])
            accent_labels.append(f["accent_labels"])
            spks.append(f["spk_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
         
        batch = {}

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        batch["uids"] = uids
        batch["durations"] = durations
        batch["accent_labels"] = accent_labels
        batch["spk_ids"] = spks
        batch["audiofiles"] = audiofiles

        return batch
