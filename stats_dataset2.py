import os
import librosa
from collections import defaultdict

def get_audio_stats(filepaths):
    """Calculate audio dataset statistics.
    
    Args:
        filepaths: List of audio file paths
        
    Returns:
        dict: {
            'total_duration_hrs': float,
            'num_speakers': int,
            'num_utterances': int,
            'avg_duration_sec': float
        }
    """
    total_duration = 0.0
    speaker_ids = set()
    
    for filepath in filepaths:
        try:
            # Get duration
            duration = librosa.get_duration(filename=filepath)
            total_duration += duration
            
            # Extract speaker ID (assuming format: speakerid_*.wav)
            filename = os.path.basename(filepath)
            # speaker_id = filename[1:5]
            # speaker_id = filename.split('_')[0]
            speaker_id = filename[:7]  # Modify based on your naming convention
            speaker_ids.add(speaker_id)
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    stats = {
        'total_duration_hrs': total_duration / 3600,
        'num_speakers': len(speaker_ids),
        'num_utterances': len(filepaths),
        'avg_duration_sec': total_duration / len(filepaths) if filepaths else 0
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Replace with your actual file list
    audio_files = []
    with open('dump2/final_labels/aishell3', 'r', encoding='utf-8') as f:
        for line in f:
            uid = line.strip().split(' ')[0]
            # audio_files.append(line.strip().split('\t')[0] + '.wav')
            
            # latic
            # spk = uid[1:5]
            # filepath = "/data2/xintong/LATIC/WAVE/WAVE/SPEAKER" + spk + '/SESSION0/' + uid + '.WAV' 


            # magichub
            # spk = uid.split('_')[3]
            # filepath = "/data2/xintong/magichub_singapore/clean_data/clean_data/wav_16k/" + spk + '/' + uid  
                
            # sichuan
            # spk = uid.split('_')[0]
            # filepath = "/data1/xintong/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence/WAV/" + spk + '/' + uid  
                
            # audio_files.append(filepath)
            spk = uid[:7]
            filepath1 = "/data2/xintong/tts_chinese/aishell3/" + "train" + '/wav_16k/' + spk + '/' + uid + '.wav'
            filepath2 = "/data2/xintong/tts_chinese/aishell3/" + "test" + '/wav_16k/' + spk + '/' + uid + '.wav'
            
            if os.path.exists(filepath1):
                audio_files.append(filepath1)
            elif os.path.exists(filepath2):
                audio_files.append(filepath2)
            else:
                print(f"File not found: {filepath1} or {filepath2}")

    stats = get_audio_stats(audio_files)
    print(f"Total duration: {stats['total_duration_hrs']:.2f} hours")
    print(f"Number of speakers: {stats['num_speakers']}")
    print(f"Number of utterances: {stats['num_utterances']}")
    print(f"Average duration: {stats['avg_duration_sec']:.2f} seconds")