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

