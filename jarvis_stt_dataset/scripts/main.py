import argparse
import os
from huggingface_hub import hf_hub_download, login, snapshot_download
import tarfile
from tqdm.auto import tqdm
from collections import defaultdict
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
import shutil
from delete_folder import delete_dir
import pandas as pd
from configs.const import const

parser = argparse.ArgumentParser(description='Create dataset')
parser.add_argument("--data_root", required=True, default=None)
parser.add_argument("--logs_path", required=False, default=None)
parser.add_argument("--text_coding", required=False, default=True)
parser.add_argument("--shuffle", required=False, default=True)
parser.add_argument("--hf_api_key", required=True, default=False)
args = parser.parse_args()

const = const()

main_path = f'{args.data_root}/{const.path2main}'
MY_FOLDER_NAMES = const.MY_FOLDER_NAMES
path2main = const.path2main
path2dist = const.path2dist
path2mc = const.path2mc
path2scripts = const.path2scripts
wait_dir = f"{main_path}/wait"

login(args.hf_api_key)

if len(os.listdir(f'{main_path}/clean')) == 1 and len(os.listdir(f'{main_path}/from_phone_home')) == 1 and len(os.listdir(f'{main_path}/from_phone_outdoors')) == 1 and len(os.listdir(f'{main_path}/in_airpods_home')) == 1 and len(os.listdir(f'{main_path}/in_airpods_outdoors')) == 1 and len(os.listdir(f'{main_path}/with_noise')) == 1:

    snapshot_download(repo_id="Ilialebedev/jarvis_stt_dataset_main", repo_type='dataset', local_dir=wait_dir)

    try:
    
        for n in MY_FOLDER_NAMES:
            shutil.copytree(f"{wait_dir}/{n}_0", f"{args.data_root}/{path2main}/{n}/{n}_0")
            
    except Exception as e:
        
        if 'File exists' in e:
            pass
        else:
            raise e

if os.path.exists(wait_dir):
    delete_dir(wait_dir)

dist_path = f'{args.data_root}/{path2dist}'

if len(os.listdir(f'{dist_path}/train')) == 1 or len(os.listdir(f'{dist_path}/validation')) == 1 or len(os.listdir(f'{dist_path}/test')) == 1:

    os.system(f'python3 {args.data_root}/{path2scripts}/process_to_manifest.py \
        --data_root={args.data_root} \
        --logs_path={args.logs_path} \
        --text_coding={args.text_coding} \
        --shuffle={args.shuffle}')
