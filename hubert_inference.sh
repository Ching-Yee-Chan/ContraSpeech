OUTPUT_DIR=/llm/nankai/zhaojiankun_space/cvss_c/ch/format_data
RESULT_PATH=/llm/nankai/zhaojiankun_space/TranSpeech/hubert_output
SUBSET_NAME=train

CUDA_VISIBLE_DEVICES=1,2,3,7 PYTHONPATH=. /llm/nankai/zhaojiankun_space/anaconda3/envs/s2st/bin/python \
   examples/speech_recognition/new/infer.py \
   --config-dir examples/hubert/config/decode/ \
   --config-name infer_viterbi \
   task.data=$OUTPUT_DIR \
   task.normalize=false \
   common_eval.results_path=$RESULT_PATH/log \
   common_eval.path=$OUTPUT_DIR/checkpoint_best.pt \
   dataset.gen_subset=$SUBSET_NAME \
   +task.labels=["unit"] \
   +decoding.results_path=$RESULT_PATH \
   common_eval.post_process=none \
   +dataset.batch_size=1 \
   common_eval.quiet=True