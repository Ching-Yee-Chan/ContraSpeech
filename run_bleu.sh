AUDIO_PATH="/home/zhaojiankun/zhaojiankun_space/TranSpeech/output/result"
REF_PATH="/home/zhaojiankun/zhaojiankun_space/TranSpeech/output/format_s2st/ref.txt"

PYTHONPATH=. python compute_asr_bleu.py --lang en \
    --audio_dirpath $AUDIO_PATH \
    --reference_path $REF_PATH \
    --reference_format txt