REF_PATH="/llm/nankai/zhaojiankun_space/cvss_c/fr/test.tsv"
DATA_PATH="/llm/nankai/zhaojiankun_space/TranSpeech/output/format_s2st/test.tsv"
RES_PATH="/llm/nankai/zhaojiankun_space/TranSpeech/output/format_s2st/ref.tsv"
text = {}

from tqdm import tqdm

with open(REF_PATH, 'r', encoding='utf-8') as manifest:
    for i, line in tqdm(enumerate(manifest)):
        line = line.strip('\n').split('\t')
        text[line[0]] = line[1]
f_write = open(RES_PATH, mode='w')
with open(DATA_PATH, 'r', encoding='utf-8') as manifest:
    for i, line in tqdm(enumerate(manifest)):
        if i == 0: continue
        line = line.strip('\n').split('\t')
        f_write.write(f"{text[line[0]]}\n")
f_write.close()
        
        