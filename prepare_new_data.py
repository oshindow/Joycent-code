import os
import re
from pydub import AudioSegment
import chardet

# Create directories if they don't exist
rootdir = '/data1/xintong/Tianjin_Dialect_Conversational_Speech_Corpus'
# Guangzhou_Cantonese_Conversational_Speech_Corpus
# Nanchang_Dialect_Conversational_Speech_Corpus
# Shanghai_Dialect_Conversational_Speech_Corpus
# Sichuan_Dialect_Conversational_Speech_Corpus 
# Tianjin_Dialect_Conversational_Speech_Corpus
# Zhengzhou_Dialect_Conversational_Speech_Corpus
os.makedirs(rootdir + '/wav_16k', exist_ok=True)

# Chinese characters and punctuation regex
chinese_regex = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+')
chinese_punctuation = '。，、；：？！“”‘’（）【】《》…—～·'

def is_chinese_only(text):
    """Check if text contains only Chinese characters and punctuation"""
    # Remove all Chinese characters and punctuation
    cleaned = chinese_regex.sub('', text)
    cleaned = ''.join([c for c in cleaned if c not in chinese_punctuation])
    # If anything remains, it's not Chinese-only
    return not cleaned.strip()

def clean_text(text):
    """Remove Chinese punctuation from text"""
    return ''.join([c for c in text if c not in chinese_punctuation]).strip()

def process_text_file(text_path, wav_dir, output_dir, content_file):
    """Process a single text file and corresponding WAV file"""
    # Detect text file encoding
    with open(text_path, 'rb') as f:
        rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding'] or 'utf-8'
    
    # Read text file
    with open(text_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(text_path))[0]
    wav_path = os.path.join(wav_dir, f"{base_name}.wav")
    
    if not os.path.exists(wav_path):
        print(f"Warning: WAV file {wav_path} not found, skipping")
        return
    
    # Load WAV file
    audio = AudioSegment.from_wav(wav_path)
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse line
        parts = re.split(r'\s+', line)
        if len(parts) < 4:
            continue
            
        time_part = parts[0]
        speaker = parts[1]
        gender = parts[2]
        text = ' '.join(parts[3:])
        
        # Skip non-Chinese segments
        if not is_chinese_only(text):
            continue
        
        # Parse time
        try:
            start_time, end_time = map(float, time_part.strip('[]').split(','))
        except:
            continue
        
        # Convert seconds to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Extract audio segment
        segment = audio[start_ms:end_ms]
        
        # Set parameters: 16k, 16bit, mono
        segment = segment.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        
        # Create output filename
        output_wav = os.path.join(output_dir, f"{base_name}_{start_time}_{end_time}.wav")
        
        # Export WAV
        segment.export(output_wav, format='wav')
        
        # Clean text (remove punctuation)
        cleaned_text = clean_text(text)
        
        # Write to content file
        content_file.write(f"{os.path.basename(output_wav)}|{cleaned_text}\n")

def main():
    text_dir = rootdir + '/TXT'
    wav_dir = rootdir + '/WAV'
    output_dir = rootdir + '/wav_16k'
    
    # Process all text files
    with open(rootdir + '/content.txt', 'w', encoding='utf-8') as content_file:
        for text_file in os.listdir(text_dir):
            if text_file.endswith('.txt'):
                text_path = os.path.join(text_dir, text_file)
                process_text_file(text_path, wav_dir, output_dir, content_file)

if __name__ == '__main__':
    main()