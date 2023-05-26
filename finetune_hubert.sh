CONFIG_DIR=/home/zhaojiankun/zhaojiankun_space/TranSpeech/examples/hubert/config/finetune
IE_MANIFEST=/home/zhaojiankun/zhaojiankun_space/cvss_c/fr/ie/manifest
SN_UNIT=/home/zhaojiankun/zhaojiankun_space/cvss_c/fr/sn/unit
CKPT=/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3.pt
LOG_DIR=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output/log
DATE_TIME=$(date +%m%d%H%M)

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. nohup python fairseq_cli/hydra_train.py \
    --config-dir $CONFIG_DIR \
  	--config-name base_10h_change \
  	task.data=$IE_MANIFEST \
	task.label_dir=$SN_UNIT \
  	model.w2v_path=$CKPT \
	optimization.max_update=70000 > ${LOG_DIR}/log_hubert_finetune_${DATE_TIME}.txt &
