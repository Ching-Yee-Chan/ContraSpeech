RESULTS_PATH=/home/zhaojiankun/zhaojiankun_space/TranSpeech/output_test/res_output_mem
GEN_SUBSET=test

grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit
