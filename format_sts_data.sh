SRC_AUDIO=/home/zhaojiankun/zhaojiankun_space/commonvoice/fr/
TGT_AUDIO=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output/hubert_output
DATA_ROOT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/format_s2st
VOCODER_CKPT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/hifigan/g_00600000
VOCODER_CFG=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/hifigan/config.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
    --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO \
    --output-root $DATA_ROOT --reduce-unit \
    --vocoder-checkpoint $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG
   