import os
import json
import torchaudio
# spk2accent = {}
with open('/home/xintong/Speech-Backbones/Grad-TTS/resources/spk2accent.json', 'r', encoding='utf8') as input:
    spk2accent = json.load(input)

spk_dict = {}
utt_dict = {}
dur_dict = {}
# audio_lengths.append(duration)
aishell3_dir = '/data2/xintong/aishell3/test/wav_16k'
for root, dirs, files in os.walk(aishell3_dir):
    for dir in dirs:
        spk = dir
        acc = spk2accent[spk]
        for file in os.listdir(os.path.join(root, dir)):
            waveform, sample_rate = torchaudio.load(os.path.join(root, dir, file))
            num_frames = waveform.size(1)
            duration = num_frames / sample_rate
        
            if acc in dur_dict:
                dur_dict[acc] += duration
            else:
                dur_dict[acc] = 0
            
            if acc in utt_dict:
                utt_dict[acc] += 1
            else:
                utt_dict[acc] = 0

            if acc in spk_dict:
                spk_dict[acc].add(spk)
            else:
                spk_dict[acc] = set()

print([len(acc) for acc in spk_dict.values()])
print(utt_dict)
print([dur / 3600 for dur in dur_dict.values()])