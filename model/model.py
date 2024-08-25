import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelConfig:
    d_model: int
    output_dim: int

    encoder_max_seq_length: int
    encoder_input_dim: int
    encoder_dim_feedforward: int
    encoder_nhead: int
    encoder_num_layers: int

    decoder_max_seq_length: int
    decoder_input_dim: int
    decoder_dim_feedforward: int
    decoder_nhead: int
    decoder_num_layers: int

    dropout: float
    activation: Literal["relu", "gelu"]
    bias: bool


class AutoregressiveTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        """
        Initialize the AutoregressiveTransformer.

        Args:
            config (ModelConfig): Configuration for the model.
            device (torch.device): Device to run the model on.
        """
        super(AutoregressiveTransformer, self).__init__()

        self.config = config
        self.device = device

        self.encoder_input_projection = nn.Linear(config.encoder_input_dim, config.d_model)
        self.decoder_input_projection = nn.Linear(config.decoder_input_dim, config.d_model)

        self.encoder_positional_encoding = LearnablePositionalEncoding(config.encoder_max_seq_length, config.d_model)
        self.decoder_positional_encoding = LearnablePositionalEncoding(config.decoder_max_seq_length, config.d_model)

        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)

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
        self.att_mask: dict[int, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        """Initialize weights of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            encoder_input (torch.Tensor): Input to the encoder.
            decoder_input (torch.Tensor): Input to the decoder.

        Returns:
            torch.Tensor: Output of the model.
        """
        encoder_output = self.encode(encoder_input)
        return self.decode(decoder_input, encoder_output)

    def encode(self, encoder_input: torch.Tensor) -> torch.Tensor:
        """Encode the input."""
        x = self.encoder_input_projection(encoder_input)
        x = self.encoder_positional_encoding(x)

        # residual connection around the encoder
        residual = x
        x = self.transformer_encoder(x)
        x = x + residual
        x = self.encoder_norm(x)

        return x

    def decode(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Decode the input."""
        x = self.decoder_input_projection(decoder_input)
        x = self.decoder_positional_encoding(x)

        _, T, _ = x.shape

        if T not in self.att_mask:
            self.att_mask[T] = self.generate_causal_mask(T)

        # residual connection around the decoder
        residual = x
        x = self.transformer_decoder(x, encoder_output, tgt_mask=self.att_mask[T])
        x = x + residual
        x = self.decoder_norm(x)

        return self.output_layer(x)

    def generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate a causal mask for the decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialize LearnablePositionalEncoding.

        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Dimension of the model.
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to the input."""
        _, T, _ = x.shape
        return x + self.positional_encoding[:, :T, :]
