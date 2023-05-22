DATA_ROOT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/format_s2st
MODEL_DIR=save_ckpt


CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. fairseq-train $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_speech_fasttranslate --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion nar_speech_to_unit --label-smoothing 0.2 \
  --arch nar_s2ut_conformer --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir $MODEL_DIR  --tensorboard-logdir $MODEL_DIR \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 1000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 20000 --max-tokens 20000 --max-target-positions 3000 --update-freq 4 \
  --seed 1 --fp16 --num-workers 8 \
  --user-dir research/  --attn-type espnet --pos-enc-type rel_pos 