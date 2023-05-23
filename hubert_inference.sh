DATA_DIR=/llm/nankai/zhaojiankun_space/cvss_c/ru/format_data
RESULT_PATH=/llm/nankai/zhaojiankun_space/TranSpeech/output/hubert_output
FINETUNE_CKPT=/llm/nankai/zhaojiankun_space/TranSpeech/output/hubert_ckpt/checkpoint_best.pt
SUBSET_NAME=dev

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. python examples/speech_recognition/new/infer.py \
   --config-dir examples/hubert/config/decode/ \
   --config-name infer_viterbi \
   task.data=$DATA_DIR \
   task.normalize=false \
   common_eval.results_path=$RESULT_PATH/log \
   common_eval.path=$FINETUNE_CKPT \
   dataset.gen_subset=$SUBSET_NAME \
   +task.labels=["unit"] \
   +decoding.results_path=$RESULT_PATH \
   common_eval.post_process=none \
   +dataset.batch_size=1 \
   common_eval.quiet=True

# DATA_DIR=/llm/nankai/zhaojiankun_space/cvss_c/ch/format_data
# RESULT_PATH=/home/zhaojiankun/zhaojiankun_space/cvss_c/ch
# FINETUNE_CKPT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3.pt
# SUBSET_NAME=test

# CUDA_VISIBLE_DEVICES=0,1,2,5 PYTHONPATH=. python examples/speech_recognition/new/infer.py \
#    --config-dir examples/hubert/config/decode/ \
#    --config-name infer_viterbi \
#    task.data=$DATA_DIR \
#    task.normalize=false \
#    common_eval.results_path=$RESULT_PATH/log \
#    common_eval.path=$FINETUNE_CKPT \
#    dataset.gen_subset=$SUBSET_NAME \
#    +task.labels=["unit"] \
#    +decoding.results_path=$RESULT_PATH \
#    common_eval.post_process=none \
#    +dataset.batch_size=1 \
#    common_eval.quiet=True