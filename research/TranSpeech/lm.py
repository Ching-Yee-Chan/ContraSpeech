import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
import copy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LanguageModel_Ru_En:
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(source='local', 
                    repo_or_dir='/home/zhaojiankun/zhaojiankun_space/TranSpeech/ckpt/wmt19.ru-en.ensemble', 
                    model='transformer.wmt19.ru-en', 
                    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt', 
                    tokenizer='moses', 
                    bpe='fastbpe')
        self.model.eval()
        self.model.cuda()

    def infer(self, sentences):
        # avoid batch division
        self.model.cfg.dataset.batch_size = 200
        original_batch_size = len(sentences)
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentences.append(self.model.encode(sentence))
            batches = self.model._build_batches(tokenized_sentences, False) #此处样本已经被打乱！
        for batch in batches:
            # batch.cuda()
            batch = utils.apply_to_sample(lambda t: t.to(self.model.device), batch)
            input = batch['net_input']
            # build generator
            gen_args = copy.deepcopy(self.model.cfg.generation)
            gen_args['beam'] = 5
            generator = self.model.task.build_generator(self.model.models, gen_args, prefix_anyowed_tokens_fn=None)
            with torch.no_grad():
                feature = generator.model.forward_encoder(input)    #torch.Size([61, 3, 1024])
            # feature now has feature vector from 4 parts
            # we only use output from the last layer
            encoder_out = feature[0]['encoder_out'][0] #torch.Size([61, 3, 1024])
            encoder_padding_mask = feature[0]['encoder_padding_mask'][0] #torch.Size([3, 61])
            # change order
            encoder_out = encoder_out.permute(1, 0, 2)  #torch.Size([3, 61, 1024])
            reorder_cache = []
            for id, hypos, mask in zip(batch["id"].tolist(), encoder_out, encoder_padding_mask):
                reorder_cache.append((id, hypos, mask))

            # sort output to match input order
            reorder_cache = sorted(reorder_cache, key=lambda x: x[0])
            encoder_out_list = [hypos for _, hypos, _ in reorder_cache]
            encoder_padding_mask_list = [mask for _, _, mask in reorder_cache]
            encoder_out = torch.stack(encoder_out_list).detach() # torch.Size([3, 61, 1024])
            encoder_padding_mask = torch.stack(encoder_padding_mask_list).detach()   # torch.Size([3, 61])
            # now feature has a dimention of [batch_size, max_len, dim=1024]
            assert encoder_out.shape[0] == original_batch_size, "Batch size inconsistent!"
            return {"encoder_out": encoder_out, "encoder_padding_mask": encoder_padding_mask}
        assert 0, "Should not reach here"
        
class LanguageModel_Fr_En:
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        model.eval()
        self.encoder = model.get_encoder()
        self.encoder.cuda()

    def infer(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True)   #{'input_ids': [B, maxlen], 'attention_mask': [B, maxlen]}
        # batch.cuda()
        device = self.encoder.device
        tokens['input_ids'] = tokens['input_ids'].to(device)
        tokens['attention_mask'] = tokens['attention_mask'].to(device)
        with torch.no_grad():
            result = self.encoder(**tokens)
        return {"encoder_out": result['last_hidden_state'], "encoder_padding_mask": 1 - tokens['attention_mask']}
        
class Adapter(nn.Module):
    def __init__(self, in_channel=256, out_channel=1024):
        super().__init__()
        self.channel_adapter = nn.Linear(in_channel, out_channel)
        
    def forward(self, speech_feature, language_feature):
        """Convert speech feature into language feature
        
        Input:
        speech_feature: [batch_size, maxlen_speech, speech_channel=256]
        language_feature: [batch_size, maxlen_language, language_channel=1024]
        
        Returns:
        language_feature_pred: [batch_size, maxlen_language, language_channel=1024]
        language_feature: [batch_size, maxlen_language, language_channel=1024]
        speech_feature_pred: [batch_size, maxlen_speech, language_channel=1024]
        speech_feature_pred: [batch_size, maxlen_speech, language_channel=1024]
        """
        # STEP1: adapt speech feature to language
        assert ~torch.isinf(speech_feature).any(), "speech_feature has INF!"
        assert ~torch.isinf(language_feature).any(), "language_feature has INF!"
        speech_feature = self.channel_adapter(speech_feature)
        # STEP2: calculate cosine similarity
        language_length = torch.linalg.vector_norm(language_feature, dim=-1, keepdim=True)
        # language_length[torch.isinf(language_length)] = 100
        assert ~torch.isinf(language_length).any(), "language_feature_norm has INF!"
        speech_length = torch.linalg.vector_norm(speech_feature, dim=-1, keepdim=True)
        # speech_length[torch.isinf(speech_length)] = 100
        assert ~torch.isinf(speech_length).any(), "speech_feature_norm has INF!"
        language_feature_norm = language_feature / language_length
        speech_feature_norm = speech_feature / speech_length
        language_feature_norm = language_feature_norm.to(speech_feature_norm.dtype)
        similarity = language_feature_norm @ speech_feature_norm.permute(0, 2, 1) #[B, maxlen_language, maxlen_speech]
        # STEP3: weighted sum
        assert ~torch.isinf(similarity).any(), "similarity has INF!"
        weight_l2s = F.softmax(similarity, dim=-1)
        assert ~torch.isinf(weight_l2s).any(), "weight_l2s has INF!"
        language_feature_pred = weight_l2s @ speech_feature #[B, maxlen_language, speech_channel->language_channel]
        
        weight_s2l = F.softmax(similarity.permute(0, 2, 1), dim=-1) #[B, maxlen_speech, maxlen_language]
        language_feature = language_feature.to(weight_s2l.dtype)
        speech_feature_pred = weight_s2l @ language_feature    #[B, maxlen_speech, language_channel]
        assert ~torch.isinf(language_feature_pred).any(), "language_feature_pred has INF!"
        return language_feature_pred, language_feature, speech_feature_pred, speech_feature