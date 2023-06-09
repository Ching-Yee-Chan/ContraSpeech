AUDIO_PATH="/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/res_output_mem"
REF_PATH="/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/res_output_mem/ref.txt"

PYTHONPATH=. python research/TranSpeech/asr_bleu/compute_asr_bleu.py --lang en \
    --audio_dirpath $AUDIO_PATH \
    --reference_path $REF_PATH \
    --reference_format txt