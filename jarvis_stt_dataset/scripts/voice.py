import librosa
import os
import pandas as pd
from pydub.audio_segment import AudioSegment
import random
import torch
# datapath = '/Users/user/Desktop/jarvis/Jarvis/jarvis_stt/jarvis_stt_dataset/jstt_dataset/main'

# old_data = pd.read_csv('/Users/user/Desktop/jarvis/Jarvis/jarvis_stt/jarvis_stt_dataset/jstt_dataset/main/in_airpods_home/main_in_airpods_home_jarvis_stt_dataset_metadata.csv', index_col=0)

# new_data = pd.read_csv('/Users/user/Desktop/jarvis/Jarvis/jarvis_stt/jarvis_stt_dataset/jstt_dataset/main/in_airpods_outdoors/main_in_airpods_outdoors_jarvis_stt_dataset_metadata.csv', index_col=0)

# for i in range(len(os.listdir(f'{datapath}/in_airpods_home/in_airpods_home_0'))):
    
#     CNT_VOICE = (len(new_data['filepath']))
    
#     text = old_data.loc[CNT_VOICE]['text']
#     duration = old_data.loc[CNT_VOICE]['duration']
    
#     cnt_voice_str = ('0'*(5-len(str(CNT_VOICE))))+str((CNT_VOICE+1))
    
#     filepath = f'{datapath}/in_airpods_outdoors/in_airpods_outdoors_0/main_in_airpods_outdoors_{cnt_voice_str}.wav'
    
#     audio, sr = librosa.load(f'{datapath}/in_airpods_home/in_airpods_home_0/main_in_airpods_home_{cnt_voice_str}.wav', sr=16000)
    
#     write(f'{datapath}/in_airpods_outdoors/in_airpods_outdoors_0/main_in_airpods_outdoors_{cnt_voice_str}.wav', sr, audio)
    
#     new_data.loc[CNT_VOICE] = filepath, text, duration
    
#     new_data.to_csv('/Users/user/Desktop/jarvis/Jarvis/jarvis_stt/jarvis_stt_dataset/jstt_dataset/main/in_airpods_outdoors/main_in_airpods_outdoors_jarvis_stt_dataset_metadata.csv')

class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        random.seed(rng) if rng else None
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for prob, p in self._pipeline:
            if random.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for prob, p in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

augmentor = AudioAugmentor()

audio = AudioSegment.from_file(
            '/Users/user/Desktop/main/from_phone_outdoors/from_phone_outdoors_0/main_from_phone_outdoors_00041.wav',
            target_sr=16000,
            int_values=False,
            offset=0,
            duration=4.1,
            trim=False,
            trim_ref=max,
            trim_top_db=60,
            trim_frame_length=2048,
            trim_hop_length=512,
            orig_sr=None,
            channel_selector=None,
            normalize_db=None,
        )

augmentor.perturb(audio)

print(torch.tensor(audio.samples, dtype=torch.float))