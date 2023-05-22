TGT_AUDIO=/llm/nankai/zhaojiankun_space/cvss_c/ru_16000/
TGT_AUDIO_SN=/llm/nankai/zhaojiankun_space/cvss_c/ru/sn

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python research/TranSpeech/hubertCTC/gen_SN.py  \
    --wav $TGT_AUDIO \
    --out $TGT_AUDIO_SN
