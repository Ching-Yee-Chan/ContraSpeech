RESULTS_PATH=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output/mini_result
GEN_SUBSET=mini_train

grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit
