import pyaudio
import wave
from random import uniform
import time
import os
import time
import logging
import pandas as pd
import librosa
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

'''

11000 - my (47,8%)
    1833 - [] (16,6%)
8000 - rus (34,8)
4000 - eng (17,4)

50 epochs

1000 voice/day

478 my_voice/day
348 rus_voice/day
174 eng_voice/day

80 folder_my_voice/day

'''

MAINDIR = '/Users/user/Desktop/jarvis/Jarvis'
SECTION = 'jarvis_stt'
SUBSECTION = 'jarvis_stt_dataset'

LOCAL_PATH = f'{MAINDIR}/{SECTION}/{SUBSECTION}'
COLAB_PATH = f'/content'

CHAPTER = 'jstt_dataset'
BRANCH = 'main'
FOLDER_NAME = 'in_airpods_home'
FILENAME = "{BRANCH}_{FOLDER_NAME}_{CNT_VOICE}.wav"

FOLDER_NAMES = ['clean', 'from_phone_home', 'from_phone_outdoors', 'in_airpods_home', 'in_airpods_outdoors', 'with_noise']

def get_path(colab, fn):
    metadata_path = f'{CHAPTER}/{BRANCH}/{fn}/{BRANCH}_{fn}_{SUBSECTION}_metadata.csv'
    return f'{CHAPTER}/{BRANCH}/{fn}/{fn}_0', f'{LOCAL_PATH}/{metadata_path}'

folder_ind = 0

def create_voice():
    
    global LOCAL_PATH, FOLDER_NAMES, folder_ind
  
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    # duration = round(uniform(1.8, 4.3), 6)

    ready_data = pd.read_csv('/Users/user/Desktop/jarvis/Jarvis/jarvis_stt/jarvis_stt_dataset/jstt_dataset/main/from_phone_outdoors/main_from_phone_outdoors_jarvis_stt_dataset_metadata.csv')

    audio_dir, metadata = get_path(False, FOLDER_NAMES[-1])

    data = pd.read_csv(metadata, index_col=0)

    CNT_VOICE = (len(data['filepath']))

    text = ready_data.loc[CNT_VOICE]['text']
    duration = round(ready_data.loc[CNT_VOICE]['duration'], 6)

    print(f"Seconds = {duration}")

    time.sleep(1)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    
    print()
    print()
    print()
    print()
    print()
    print()
    print("Recording...")
    print()
    print()
    print()
    print()
    print()
    print()
    
    for i in range(int(sample_rate / chunk * duration)):
        sdata = stream.read(chunk)
        frames.append(sdata)
        
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    cnt_voice_str = ('0'*(5-len(str(CNT_VOICE))))+str((CNT_VOICE+1))
     
    audio_filepath = os.path.join(audio_dir, FILENAME.format(BRANCH=BRANCH, FOLDER_NAME=FOLDER_NAME, CNT_VOICE=cnt_voice_str))

    wf = wave.open(os.path.join(LOCAL_PATH, audio_filepath), "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    data.loc[CNT_VOICE] = audio_filepath, text, duration
    
    data.to_csv(metadata)
    
    time.sleep(2)
    
if __name__ == '__main__':
    while folder_ind < 6:
        create_voice()

