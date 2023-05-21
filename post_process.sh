RESULTS_PATH=/llm/nankai/zhaojiankun_space/TranSpeech/hubert_output
DATA_DIR=/llm/nankai/zhaojiankun_space/cvss_c/ch/format_data

SUBSET_NAME=train
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
    --in-unit $RESULTS_PATH/hypo.units \
    --in-audio $DATA_DIR/$SUBSET_NAME.tsv \
    --output-root ${RESULTS_PATH}

SUBSET_NAME=test
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
    --in-unit $RESULTS_PATH/hypo.units \
    --in-audio $DATA_DIR/$SUBSET_NAME.tsv \
    --output-root ${RESULTS_PATH}

SUBSET_NAME=dev
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
    --in-unit $RESULTS_PATH/hypo.units \
    --in-audio $DATA_DIR/$SUBSET_NAME.tsv \
    --output-root ${RESULTS_PATH}
 
