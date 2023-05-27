RESULTS_PATH=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output/mini_result
GEN_SUBSET=mini_train
VOCODER_CKPT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/vocoder/g_00500000
VOCODER_CFG=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/vocoder/config.json

CUDA_VISIBLE_DEVICES=0,1,2,5 PYTHONPATH=. python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${RESULTS_PATH} --dur-prediction
  # --results-path ${RESULTS_PATH} 
