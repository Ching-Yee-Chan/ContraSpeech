DATA_ROOT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/format_s2st
MODEL_DIR=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/s2ut_ckpt
LOG_DIR=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/log
DATE_TIME=$(date +%m%d%H%M)

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. fairseq-train $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_speech_fasttranslate --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion nar_speech_to_unit --label-smoothing 0.2 \
  --arch nar_s2ut_conformer --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir $MODEL_DIR  --tensorboard-logdir $MODEL_DIR \
  --lr 0.000125 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 2500 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 400000 --max-tokens 20000 --max-target-positions 3000 --update-freq 4 \
  --seed 1 --fp16 --num-workers 8 \
  --user-dir research/  --attn-type espnet --pos-enc-type rel_pos 
  > ${LOG_DIR}/log_s2ut_train_${DATE_TIME}.txt &
