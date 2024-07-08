import torch
import torch.nn as nn
# note that torchtune requires setuptools' version to be < 70
from torchtune.modules import RotaryPositionalEmbeddings

from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int

    d_model: int
    max_sequence_length: int

    encoder_dim_feedforward: int
    encoder_nhead: int
    encoder_num_layers: int

    decoder_dim_feedforward: int
    decoder_nhead: int
    decoder_num_layers: int

    dropout: float
    activation: str
    bias: bool


class AutoregressiveTransformer(nn.Module):

    def __init__(self, config, device):
        super(AutoregressiveTransformer, self).__init__()

        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.rotary_positional_embedding = RotaryPositionalEmbeddings(
            config.d_model, max_seq_len=config.max_sequence_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_d_model,
            nhead=config.encoder_nhead,
            dim_feedforward=config.encoder_dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
            bias=config.bias,
            device=device,
            rotary_pos_emb=self.rotary_positional_embedding,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_d_model,
            nhead=config.decoder_nhead,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
            bias=config.bias,
            device=device,
            rotary_pos_emb=self.rotary_positional_embedding,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_num_layers)

        self.output_layer = nn.Linear(config.d_model, config.output_dim)
        self.att_mask = {}

    def forward(self, x, device):
        x = self.input_projection(x)

        seq_len = x.size(1)
        if seq_len not in self.att_mask:
            self.att_mask[seq_len] = self.generate_causal_mask(seq_len, device)

        encoder_output = self.transformer_encoder(x)
        output = self.transformer_decoder(x, encoder_output, tgt_mask=self.att_mask[seq_len])
        output = self.output_layer(output)

    def generate_causal_mask(seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class TransformerEncoderLayerWithRoPE(nn.TransformerEncoderLayer):
    def __init__(self, *args, rotary_pos_emb=None, **kwargs):
        super(TransformerEncoderLayerWithRoPE, self).__init__(*args, **kwargs)
        self.rotary_pos_emb = rotary_pos_emb

    def forward(self, src, *args, **kwargs):
        src = self.rotary_pos_emb(src)
        return super(TransformerEncoderLayerWithRoPE, self).forward(src, *args, **kwargs)


class TransformerDecoderLayerWithRoPE(nn.TransformerDecoderLayer):
    def __init__(self, *args, rotary_pos_emb=None, **kwargs):
        super(TransformerDecoderLayerWithRoPE, self).__init__(*args, **kwargs)
        self.rotary_pos_emb = rotary_pos_emb

    def forward(self, tgt, memory, *args, **kwargs):
        tgt = self.rotary_pos_emb(tgt)
        return super(TransformerDecoderLayerWithRoPE, self).forward(tgt, memory, *args, **kwargs)
