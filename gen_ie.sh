CKPT=/llm/nankai/zhaojiankun_space/TranSpeech/ckpt/hifigan/g_00600000

TGT_AUDIO=/llm/nankai/zhaojiankun_space/cvss_c/ru_16000/train
TGT_AUDIO_IE=/llm/nankai/zhaojiankun_space/cvss_c/ru/ie/train
CUDA_VISIBLE_DEVICES=0,1,3,5,7 PYTHONPATH=. python research/TranSpeech/hubertCTC/gen_IE.py \
    --ckpt $CKPT \
    --wav $TGT_AUDIO \
    --out $TGT_AUDIO_IE
    
TGT_AUDIO=/llm/nankai/zhaojiankun_space/cvss_c/ru_16000/test
TGT_AUDIO_IE=/llm/nankai/zhaojiankun_space/cvss_c/ru/ie/test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. python research/TranSpeech/hubertCTC/gen_IE.py \
    --ckpt $CKPT \
    --wav $TGT_AUDIO \
    --out $TGT_AUDIO_IE

TGT_AUDIO=/llm/nankai/zhaojiankun_space/cvss_c/ru_16000/dev
TGT_AUDIO_IE=/llm/nankai/zhaojiankun_space/cvss_c/ru/ie/dev
CUDA_VISIBLE_DEVICES=7,5,3,1,0,2,4,6 PYTHONPATH=. python research/TranSpeech/hubertCTC/gen_IE.py \
    --ckpt $CKPT \
    --wav $TGT_AUDIO \
    --out $TGT_AUDIO_IE
