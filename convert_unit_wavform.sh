RESULTS_PATH=/home/zhaojiankun/zhaojiankun_space/TranSpeech/res_output
GEN_SUBSET=test

grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit
