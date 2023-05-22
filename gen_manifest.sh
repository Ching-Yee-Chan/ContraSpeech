DIR=/llm/nankai/zhaojiankun_space/cvss_c/fr_16000/test
EXT=wav
VALID=0

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/wav2vec/wav2vec_manifest.py \
    $DIR \
    --dest $DIR/manifest \
    --ext $EXT \
    --valid-percent $VALID
