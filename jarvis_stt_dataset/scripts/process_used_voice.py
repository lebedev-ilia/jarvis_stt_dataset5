import yaml
import os
import datetime
import pickle
import json
from configs.const import const
import shutil

const = const()

path2config = const.path2config

def logging_used_voice(logs_path : str, data_dir : str, ratio_for_next_iter):

    if logs_path is None:
        
        logs_path = data_dir + 'jarvis_stt_dataset/jstt_dataset'
    
    with open(f'{data_dir}/{path2config}') as f:
        data = yaml.safe_load(f).get('model')

    manifest_data = [data.get('train_ds').get('manifest_filepath'), data.get('validation_ds').get('manifest_filepath')]

    ids = {'train':[],'validation':[]}

    i = 0

    for m in manifest_data:
        with open(m, 'r') as f:
            m = os.path.split(m)[1]
            key = m[m.index('stt')+4:m.index('_manifest')]
            for el in f:
                ids[key].append(os.path.split(el[el.index('audio_filepath')+17:el.index('duration')-4])[1])
                i += 1

    date = str(datetime.datetime.now())[:-7]
    
    if not os.path.exists(f'{logs_path}/logs'):
        os.mkdir(f'{logs_path}/logs')
        os.mkdir(f'{logs_path}/logs/{date}')
        with open(f'{logs_path}/logs/{date}/ids_log.p', 'a'):
            pass
        with open(f'{logs_path}/logs/{date}/ratio_for_next_iter.json', 'a'):
            pass
        
    with open(f'{logs_path}/logs/{date}/ids_log.p', 'wb') as f:      
        pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{logs_path}/logs/{date}/ratio_for_next_iter.json', 'w') as f:      
        json.dump(ratio_for_next_iter, f)

    print(f'Logging {i} used voice to ---| {logs_path}/logs/{date} |---')
        
        
def get_used_voice(datadir, logs_path):
    
    if logs_path is not None:

      datadir = logs_path

    else:

      datadir = datadir + 'jarvis_stt_dataset/jstt_dataset/logs'

    if os.path.exists(f'{datadir}/logs'):

      log_data = os.listdir(f'{datadir}/logs')
      
      if len(log_data) == 0:

          print('Empty logs')
          
      else:
          
          if os.path.isfile(f'{datadir}/logs/empty.empty'):
              
              os.remove(f'{datadir}/logs/empty.empty')
          
          full_path_list = []    
          for log in log_data:

              if '2025' in log:
              
                with open(f'{datadir}/logs/{log}/ids_log.p', 'rb') as f:
                    data = pickle.load(f)
                    for k in data.keys():
                        
                        voice_list = data.get(k)
                        
                        for el in voice_list:
                            if 'main' in el:
                                name = el[el.index('main')+5:el.index('.wav')-6]
                                if name[-1] == '_':
                                    name[-1].replace('_', '')
                            
                            full_path_list.append(el)

          return list(set(full_path_list))

    else:

      print('Empty logs')
