DIR=/llm/nankai/zhaojiankun_space/cvss_c/ru/ie
EXT=wav
VALID=0.01

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/wav2vec/wav2vec_manifest.py \
    $DIR \
    --dest $DIR/manifest \
    --ext $EXT \
    --valid-percent $VALID
