import torch
import torch.nn as nn

# note that torchtune requires setuptools' version to be < 70
from torchtune.modules import RotaryPositionalEmbeddings

from dataclasses import dataclass


@dataclass
class ModelConfig:
    max_sequence_length: int
    d_model: int
    output_dim: int

    encoder_input_dim: int
    encoder_dim_feedforward: int
    encoder_nhead: int
    encoder_num_layers: int

    decoder_input_dim: int
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

        self.encoder_input_projection = nn.Linear(config.encoder_input_dim, config.d_model)
        self.decoder_input_projection = nn.Linear(config.decoder_input_dim, config.d_model)

        self.positional_encoding = LearnablePositionalEncoding(config.max_sequence_length, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_nhead,
            dim_feedforward=config.encoder_dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
            bias=config.bias,
            device=device,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_nhead,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
            bias=config.bias,
            device=device,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_num_layers)

        self.output_layer = nn.Linear(config.d_model, config.output_dim)
        self.att_mask = {}

    def forward(self, encoder_input, decoder_input, device):
        encoder_output = self.encode(encoder_input)
        return self.decode(decoder_input, encoder_output, device)

    def encode(self, encoder_input):
        x = self.encoder_input_projection(encoder_input)
        x = self.positional_encoding(x)
        return self.transformer_encoder(x)

    def decode(self, decoder_input, encoder_output, device):
        x = self.decoder_input_projection(decoder_input)
        x = self.positional_encoding(x)

        _, T, _ = x.shape

        if T not in self.att_mask:
            self.att_mask[T] = self.generate_causal_mask(T, device)

        output = self.transformer_decoder(x, encoder_output, tgt_mask=self.att_mask[T])
        return self.output_layer(output)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

    def forward(self, x):
        _, T, _ = x.shape
        return x + self.positional_encoding[:, :T, :]


# i want to try this out later, but proper rope is applied to q and k and not tgt or src
# thus i need to implement the transformer by myself
# stay simple first, then improve
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
