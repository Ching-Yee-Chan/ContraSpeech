from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class SemanticMemoryLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        cfg (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, embed_dim=512, fn_hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.fn_hidden_dim = fn_hidden_dim
        # self.quant_noise = cfg.quant_noise.pq
        # self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim)
        # self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn("relu")
        activation_dropout_p = dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = True
        self.fc1 = self.build_fc1(
            self.embed_dim,
            self.fn_hidden_dim
        )
        self.fc2 = self.build_fc2(
            self.fn_hidden_dim,
            self.embed_dim
        )

        # self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim):
        # return quant_noise(
        #     nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        # )
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        # return quant_noise(
        #     nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        # )
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim):
        return MultiheadAttention(
            embed_dim,
            num_heads=8,
            dropout=0.1,
            self_attention=True
            # q_noise=self.quant_noise,
            # qn_block_size=self.quant_noise_block_size,
            # xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        q,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            q (Tensor): query memory to the layer of shape '(seq_len, batch, embed_dim)' 
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = q
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
            q = self.self_attn_layer_norm(q)
        q, _ = self.self_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        q = self.dropout_module(q)
        q = self.residual_connection(q, residual)
        if not self.normalize_before:
            q = self.self_attn_layer_norm(q)

        residual = q
        if self.normalize_before:
            q = self.final_layer_norm(q)
        q = self.activation_fn(self.fc1(q))
        q = self.activation_dropout_module(q)
        q = self.fc2(q)


        q = self.dropout_module(q)
        q = self.residual_connection(q, residual)
        if not self.normalize_before:
            q = self.final_layer_norm(q)

        return q
    
class SemanticAdapter(nn.Module):
    def __init__(self, memory_size, dim, num_layers):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        self.memory = nn.Embedding(self.memory_size, self.dim)
        self.layers = nn.ModuleList([SemanticMemoryLayer() for _ in range(num_layers)])
    
    def forward(self, x, padding_mask=None):
        """ Generate text and audio consistent feature

        Args:
            x (Tensor): the input text or audio feature of shape '(seq_len, batch, embed_dim)'
            padding_mask (Tensor): padding mask of the input of shape '(batch, seq_len)'

        Returns:
            sematntic_memory: the modality consistent feature of shape '(seq_len, batch, embed_dim)'
        """
        _, B, _ = x.shape
        # x = x.permute(1, 0, 2)
        query = self.memory.weight.unsqueeze(1).repeat(1, B, 1)
        for layer in self.layers:
            query = layer(x, query, padding_mask)
        sematntic_memory = query
        return sematntic_memory
        
        
    
