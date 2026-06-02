import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import torchaudio

def get_audio_lengths(folder_path):
    audio_lengths = []

    # Traverse all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an audio file
            if file.endswith(('.mp3', '.wav', '.flac', '.aac')):
                file_path = os.path.join(root, file)
                waveform, sample_rate = torchaudio.load(file_path)

                # 获取音频的帧数（即采样点数）
                num_frames = waveform.size(1)

                # 计算音频时长
                duration = num_frames / sample_rate
                audio_lengths.append(duration)

    return audio_lengths

def plot_audio_lengths(audio_lengths):
    plt.hist(audio_lengths, bins=30, edgecolor='black')
    plt.title('Distribution of Audio Lengths')
    plt.xlabel('Length (seconds)')
    plt.ylabel('Number of Files')
    plt.savefig('LJSpeech.png')

# Specify the folder path
folder_path = '/data1/xintong/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence/WAV'

# Get audio lengths
audio_lengths = get_audio_lengths(folder_path)
# print([(length[0], length[1] / 60) for length in audio_lengths])
print(sum(audio_lengths) / 3600)
# print(len(audio_lengths), audio_lengths)
# Plot the lengths
# plot_audio_lengths(audio_lengths)


