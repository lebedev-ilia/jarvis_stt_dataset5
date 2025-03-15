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

tar_path = f'{main_path}/en'

if not os.path.exists(f'{tar_path}/en_dev_0') or len(os.listdir(f'{tar_path}/en_dev_0')) == 0:

    hf_hub_download(repo_id="Ilialebedev/jarvis_stt_dataset_en_tar", filename="en_dev_0.tar", repo_type="dataset", local_dir=tar_path)
    
    tar = tarfile.open(f"{tar_path}/en_dev_0.tar", "r")
    tar.extractall(f"{main_path}/en")

    os.remove(f'{main_path}/en/en_dev_0.tar')
    
    path_to_voice = f"{main_path}/en"
    
    new_data_path = f'{path_to_voice}/en_0.tsv'
    
    data = pd.read_csv(new_data_path, sep='\t', index_col='sentence_id')
    
    paths = [path for path in map(lambda x: os.path.join(path_to_voice, f'/en_dev_0/', x.path), data.iloc(0))]
    path = f'{main_path}/en/en_dev_0'
    remove_tqdm = tqdm(os.listdir(path), desc='Deleting unused voice...', leave=True)
    for item in remove_tqdm:
        new_item = os.path.join('/en_dev_0/', item)
        if new_item not in paths:
            os.remove(f'{path_to_voice}{new_item}')

if 'mozilla-foundation' not in os.listdir(f'{main_path}/rus'):

    os.system(f'python3 {args.data_root}/jarvis_stt_dataset/scripts/convert_hf_dataset_to_nemo.py \
        output_dir="{main_path}/rus" \
        path={path2mc} \
        name="ru" \
        split="validation" \
        ensure_ascii=False \
        use_auth_token=True')

dist_path = f'{args.data_root}/{path2dist}'

print(len(os.listdir(f'{dist_path}/test')) == 1)

if len(os.listdir(f'{dist_path}/train')) == 1 or len(os.listdir(f'{dist_path}/validation')) == 1 or len(os.listdir(f'{dist_path}/test')) == 1:
    
    os.system(f'python3 {args.data_root}/{path2scripts}/process_to_manifest.py \
        --data_root={args.data_root} \
        --main_ratio={args.main_ratio} \
        --dist_ratio={args.dist_ratio} \
        --shuffle={args.shuffle}')

if 'tokenizer' not in os.listdir(f'{args.data_root}/jarvis_stt_dataset/jstt_dataset'):
    
    MANIFEST_CLEANED = f'{args.data_root}/{path2dist}'
    VAL_MANIFEST_CLEANED = f"{MANIFEST_CLEANED}/validation/jarvis_stt_validation_manifest.json"
    TRAIN_MANIFEST_CLEANED = f"{MANIFEST_CLEANED}/train/jarvis_stt_train_manifest.json"
    TOKENIZER_DIR = f"{args.data_root}/jarvis_stt_dataset/jstt_dataset/tokenizer"
    
    def get_charset(manifest_data):
        charset = defaultdict(int)
        for row in tqdm(manifest_data, desc="Computing character set"):
            text = row['text']
            for character in text:
                charset[character] += 1
        return charset

    val_manifest_data = read_manifest(VAL_MANIFEST_CLEANED)
    train_manifest_data = read_manifest(TRAIN_MANIFEST_CLEANED)

    val_charset = get_charset(val_manifest_data)
    train_charset = get_charset(train_manifest_data)

    train_val_set = set.union(set(val_charset.keys()), set(train_charset.keys()))

    VOCAB_SIZE = len(train_val_set) + 2

    os.system(f'python3 {args.data_root}/{path2scripts}/process_asr_text_tokenizer.py \
    --manifest={VAL_MANIFEST_CLEANED},{TRAIN_MANIFEST_CLEANED} \
    --vocab_size={VOCAB_SIZE} \
    --data_root={TOKENIZER_DIR} \
    --tokenizer="spe" \
    --spe_type="bpe" \
    --spe_character_coverage=1.0 \
    --no_lower_case')


