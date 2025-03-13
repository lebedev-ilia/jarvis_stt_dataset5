import pandas as pd
import os
import math
from random import randint, shuffle
from tqdm import tqdm
from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import json
from process_used_voice import logging_used_voice, get_used_voice
from librosa import get_duration
from configs.const import const


const = const()

path2main = const.path2main
path2logs = const.path2logs
path2dist = const.path2dist
path2mc = const.path2mc
FOLDER_NAMES = const.FOLDER_NAMES
DISTRIBUTES = const.DISTRIBUTES


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    write_manifest(filepath, data)
    return filepath


def load_data_csv(data_path, dist, full_path_list: None):

    if dist == 'en':
        
        path_to_voice = f"{path2main}/{dist}"
        
        new_data_path = f'{data_path}/{path_to_voice}/{dist}_0.tsv'

        data = pd.read_csv(new_data_path, sep='\t', index_col='sentence_id')
        
        if full_path_list:
            
            for i, el in enumerate(data.iloc(0)):
                
                if el.path in full_path_list:
                    
                    data = data.drop([el.name])
        
        paths = []
        texts = []
        durations = []

        paths = [path for path in map(lambda x: os.path.join(data_path, path_to_voice, f'{dist}_dev_0/', x.path), data.iloc(0))]

        for el in data.iloc(0):
            ord_str = ""
            for i in el.sentence:
                if i not in sym:
                    i = rf"\u{ord(i):04x}"
                ord_str += i
            texts.append(ord_str)
            
        durations = [duration for duration in map(lambda x: round(get_duration(filename=x), 4), paths)]
        
    else:
        
        new_data_path = f'{data_path}/{path2main}/{dist}/main_{dist}_jarvis_stt_dataset_metadata.csv'
        
        data = pd.read_csv(new_data_path, index_col=0)
        
        if full_path_list:
            
            for i, el in enumerate(data.iloc(0)):
                
                if os.path.split(el.filepath)[1] in full_path_list:

                    data = data.drop([i])
        
        paths = [path for path in map(lambda x: os.path.join(data_path, 'jarvis_stt_dataset', x), list(data['filepath']))]

        sym = [' ', ',', '?', '-', '.', '’', '!']

        texts = []

        for string in list(data['text']):
            ord_str = ""
            for i in string:
                if i not in sym:
                    i = rf"\u{ord(i):04x}"
                ord_str += i
            texts.append(ord_str)
            
        durations = [duration for duration in map(lambda x: round(x, 4), list(data['duration']))]
    
    return texts, paths, durations


def load_data_json(data_path):
    
    texts =[]
    paths = []
    durations = []
    
    with open(data_path, 'r') as f:
        for s in f:
            texts.append(s[s.index('text')+8:s.index('up_votes')-4])
            paths.append(s[s.index('audio_filepath')+18:s.index('duration')-4])
            durations.append(s[s.index('duration')+11:-2])
    return texts, paths, durations


def process_to_manifest(data_path, main_ratio, dist_ratio, _shuffle):
    
    main_nums = []
    
    nums_names = {
            'clean': 0,
            'en': 0,
            'from_phone_home': 0,
            'from_phone_outdoors': 0,
            'in_airpods_home': 0,
            'in_airpods_outdoors': 0,
            'rus': 0,
            'with_noise': 0
            }
    
    if len(os.listdir(f'{data_path}/{path2logs}')) != 0:
        
        full_path_list = get_used_voice(data_path)
        
    else:
        
        full_path_list = None
    
    ratio_for_next_iter = {}
    
    for i, n in enumerate(FOLDER_NAMES):
        
        if n == 'rus':
            
            if full_path_list:
                
                num = len([path for path in os.listdir(f'{data_path}/{path2main}/{n}/{path2mc}/{n[:-1]}/validation/{n[:-1]}_dev_0') if path not in full_path_list])
                
            else:
                
                num = len([path for path in os.listdir(f'{data_path}/{path2main}/{n}/{path2mc}/{n[:-1]}/validation/{n[:-1]}_dev_0')])
                
        elif n == 'en':
            
            if full_path_list:
                
                num = len([path for path in os.listdir(f'{data_path}/{path2main}/{n}/{n}_dev_0') if path not in full_path_list])
                
            else:
                
                num = len([path for path in os.listdir(f'{data_path}/{path2main}/{n}/{n}_dev_0')])
                
        else:
            
            if full_path_list:
                
                num = len([path for path in os.listdir(f'{data_path}/{path2main}/{n}/{n}_0') if path not in full_path_list])
            else:
                
                num = len([path for path in os.listdir(f'{data_path}/{path2main}/{n}/{n}_0')])
                
        nums_names[n] = [i for i in range(num)]
        main_nums.append(math.floor(num * (main_ratio[i] / 100)))
        
        chapter = math.floor(num * (main_ratio[i] / 100))
        Available_voice_for_nex_iter = num - math.floor(num * (main_ratio[i] / 100))

        if Available_voice_for_nex_iter < 2:
            
            r = 100 / (chapter / 80)
        
        else:
            
            r = main_ratio[i] * (chapter / (Available_voice_for_nex_iter * (main_ratio[i] / 100)))
        
        ratio_for_next_iter[n] = r
        
    dist_nums = []    
    
    for i in range(len(DISTRIBUTES)):
        ratio = (dist_ratio[i] / 100)
        dist_nums_ratio = []
        for k in main_nums:
            dist_nums_ratio.append(math.floor(k * ratio))
        dist_nums.append(dist_nums_ratio)
        
    for idx, d in enumerate(DISTRIBUTES):
        
        manifest_path = f"{data_path}/{path2dist}/{d}/jarvis_stt_{d}_manifest.json"
        
        convert_tqdm = tqdm(FOLDER_NAMES, desc=f'Converting to {d}...', leave=True)
        
        with open(manifest_path, 'w') as f:
            
            for ind, n in enumerate(convert_tqdm):
                
                if n == 'rus':
                    
                    val_manifest = f"{data_path}/{path2main}/{n}/{path2mc}/{n[:-1]}/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"

                    val_data = read_manifest(val_manifest)

                    val_manifest_cleaned = write_processed_manifest(val_data, val_manifest)
                    
                    texts, paths, durations = load_data_json(val_manifest_cleaned)
                else:
        
                    texts, paths, durations = load_data_csv(data_path, n, full_path_list)   
                        
                if _shuffle:
                    
                    rand_list = []
                    
                    rand_flag = []
                    
                    for i in range(len(nums_names.get(n))):
                        if i >= dist_nums[idx][ind]:
                            rand_flag.append(False)
                        else:
                            rand_flag.append(True)
                    
                    shuffle(rand_flag)
                    
                    for i, el in enumerate(nums_names.get(n)):
                        if rand_flag[i] == True:
                            rand_list.append(el)
                    
                    for i in range(len(rand_list)):
                        
                        f.write(f'''{{"text":"{texts[rand_list[i]]}", "audio_filepath":"{paths[rand_list[i]]}", "duration":{durations[rand_list[i]]}}}\n''')
                        
                    nums_names[n] = [i for i in range(len(nums_names.get(n))) if i not in rand_list]
                    
                else:
                    
                    for i in range(len(paths)):
                        
                        f.write(f'''{{"text":"{texts[i]}", "audio_filepath":"{paths[i]}", "duration":{durations[i]}}}\n''')
                        
    logging_used_voice(data_path, ratio_for_next_iter)
    
        
        
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument("--data_root", required=True, default=None, help='data file from which to create tokenizer model')
    parser.add_argument("--shuffle", required=False, default=False, help='data file from which to create tokenizer model')
    args = parser.parse_args()
    
    data_path = args.data_root
    main_ratio = const.main_ratio
    dist_ratio = const.dist_ratio
    _shuffle = args.shuffle
    
    process_to_manifest(data_path=data_path, main_ratio=main_ratio, dist_ratio=dist_ratio, _shuffle=_shuffle)
