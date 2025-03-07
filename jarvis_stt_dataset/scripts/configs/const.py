class const():
    
    mainpath = "/Users/user/Desktop/exp/dataset"
    
    path2config = "jarvis_stt_dataset/scripts/configs/fastconformer_hybrid_transducer_ctc_bpe_colab.yaml"
    path2main = "jarvis_stt_dataset/jstt_dataset/main"
    path2logs = "jarvis_stt_dataset/jstt_dataset/logs"
    path2dist = "jarvis_stt_dataset/jstt_dataset/distributed"
    path2mc = "mozilla-foundation/common_voice_17_0"
    path2scripts = "jarvis_stt_dataset/scripts"
    
    FOLDER_NAMES = ['clean', 'en', 'from_phone_home', 'from_phone_outdoors', 'in_airpods_home', 'in_airpods_outdoors', 'rus', 'with_noise']
    MY_FOLDER_NAMES = ['clean', 'from_phone_home', 'from_phone_outdoors', 'in_airpods_home', 'in_airpods_outdoors', 'with_noise']
    DISTRIBUTES = ['train', 'validation']
    
    main_ratio = (100, 4.525, 100, 100, 100, 100, 3.65, 100)
    dist_ratio = (70, 30)