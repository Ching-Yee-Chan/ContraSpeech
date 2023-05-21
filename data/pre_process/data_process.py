import os
from tqdm import tqdm
from ffmpy import FFmpeg as mpy

def read_folder(mp3_folder, wav_folder, manifest_name):
    with open(manifest_name, 'r', encoding='utf-8') as manifest:
        cnt = 0
        for line in tqdm(manifest):
            if cnt != 0:
                line = line.strip('\n').split('\t')
                file_name = line[1]
                mp3_file = os.path.join(mp3_folder, file_name)
                trans_to_wav(mp3_file, wav_folder)
            cnt = cnt + 1


def trans_to_wav(mp3_file, wav_folder):
    file_fmt = os.path.basename(mp3_file).strip()
    file_fmt = file_fmt.split('.')[-1]
    if not os.path.exists(wav_folder):
        os.mkdir(wav_folder)
    wav_file_path = os.path.join(wav_folder, mp3_file.split('/')[-1] + '.wav')
    cmder = '-f wav -ac 1 -ar 16000'
    mpy_obj = mpy(
        inputs={
            mp3_file: None
        },
        outputs={
            wav_file_path: cmder
        }
    )
    mpy_obj.run()
    
if __name__ == '__main__':
    mp3_folder = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/clips'
    train_manifest = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/train.tsv'
    test_manifest = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/test.tsv'
    dev_manifest = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/dev.tsv'
    train_folder = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/train'
    test_folder = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/test'
    dev_folder = '/home/zhaojiankun/zhaojiankun_space/commonvoice/zh/dev'
    read_folder(mp3_folder, train_folder, train_manifest)
    read_folder(mp3_folder, test_folder, test_manifest)
    read_folder(mp3_folder, dev_folder, dev_manifest)