MANIFEST=/home/zhaojiankun/zhaojiankun_space/cvss_c/fr_16000/test/manifest/train.tsv
OUT_QUANTIZED_FILE=/home/zhaojiankun/zhaojiankun_space/cvss_c/fr_16000/test_quantized.unit
KM_MODEL_PATH=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
CKPT_PATH=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3.pt

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer 11 \
    --manifest_path $MANIFEST  \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".wav"
