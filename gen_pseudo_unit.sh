MANIFEST=/llm/nankai/zhaojiankun_space/cvss_c/fr/sn/manifest
QUANTIZED=/llm/nankai/zhaojiankun_space/cvss_c/fr/sn/quantized
UNIT=/llm/nankai/zhaojiankun_space/cvss_c/fr/sn/unit

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python data/huberts/generate_tunehuberts.py \
    --manifest $MANIFEST \
    --txt $QUANTIZED \
    --unit $UNIT
