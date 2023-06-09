DATA_ROOT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/format_s2st
MODEL_DIR=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/s2ut_ckpt/checkpoint_best.pt
RES_OUTPUT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/res_output_mem
ITER=5

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. fairseq-generate $DATA_ROOT \
    --gen-subset test --task speech_to_speech_fasttranslate  --path $MODEL_DIR \
    --target-is-code --target-code-size 1000 --vocoder code_hifigan   --results-path $RES_OUTPUT \
    --iter-decode-max-iter $ITER  --iter-decode-eos-penalty 0 --beam 1   --iter-decode-with-beam 15 \
    --user-dir research/
