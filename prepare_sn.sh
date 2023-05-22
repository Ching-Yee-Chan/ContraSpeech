AUIDO_EXT=wav
GEN_SUBSET=dev
AUDIO_DIR=/home/zhaojiankun/zhaojiankun_space/cvss_c/fr_16000
DATA_DIR=/home/zhaojiankun/zhaojiankun_space/cvss_c/fr/format_data

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. python examples/speech_to_speech/preprocessing/prep_sn_data.py \
  --audio-dir $AUDIO_DIR --ext $AUIDO_EXT \
  --data-name $GEN_SUBSET --output-dir $DATA_DIR \
  --for-inference
