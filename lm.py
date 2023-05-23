import torch
from fairseq import utils
import copy

lm = torch.hub.load(source='local', 
                    repo_or_dir='/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/wmt19.ru-en.ensemble', 
                    model='transformer.wmt19.ru-en', 
                    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt', 
                    tokenizer='moses', 
                    bpe='fastbpe')
lm.eval()
lm.cuda()

sentences = ['Привет', 
             'Где родина, там и маяк веры', 
             'Небо душное, облака смешанные, наполовину холодные, наполовину горячие, \
                ветер прилипает, дождь горький, наполовину усталый, наполовину жаждущий, \
                я слышу Спроси счастье, почему ты игнорируешь взлеты и падения',
            ]
tokenized_sentences = []
for sentence in sentences:
    tokenized_sentences.append(lm.encode('Привет'))
batches = lm._build_batches(tokenized_sentences, False) #此处样本已经被打乱！
for batch in batches:
    # batch.cuda()
    batch = utils.apply_to_sample(lambda t: t.to(lm.device), batch)
    input = batch['net_input']
    # build generator
    gen_args = copy.deepcopy(lm.cfg.generation)
    gen_args['beam'] = 5
    generator = lm.task.build_generator(lm.models, gen_args, prefix_allowed_tokens_fn=None)
    feature = generator.model.forward_encoder(input)    #torch.Size([61, 3, 1024])
    # feature now has feature vector from 4 parts
    feature_single = feature[0]['encoder_out'][0]
    # now feature has a dimention of [max_len, batch_size, dim=1024]
    print(feature_single)