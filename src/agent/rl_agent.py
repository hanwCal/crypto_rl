import torch
import torch.nn as nn
import torch.nn.functional as F


class BenchmarkTransformer(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward, dropout):
        super(BenchmarkTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, 16)

        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)
        x = x[-1, :, :]

        x = self.fc(x)

        return x

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward, dropout):
        super(AttentionMLP, self).__init__()

        # Input projection to match the expected dimension for the attention layer
        self.input_projection = nn.Linear(input_dim, dim_feedforward)

        # Multi-head attention mechanism
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dim_feedforward, num_heads=nhead, dropout=dropout)

        # Residual Network (ResNet) layers
        self.residual_blocks = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # MLP Decoder to output confidence score
        self.mlp_decoder = nn.Sequential(
            nn.Linear(dim_feedforward, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # Shape: (batch_size, seq_length, dim_feedforward)
        x = x.permute(1, 0, 2)  # Permute for compatibility with MultiheadAttention: (seq_length, batch_size, dim_feedforward)

        # Multi-head attention
        attn_output, _ = self.multihead_attention(x, x, x)  # Self-attention
        x = x + self.dropout(attn_output)  # Add & Normalize (Residual Connection)

        # Residual blocks
        residual_output = self.residual_blocks(x)
        x = x + residual_output  # Add & Normalize (Residual Connection)

        # Extract features from the last time step
        x = x[-1, :, :]  # Shape: (batch_size, dim_feedforward)

        # MLP Decoder
        output = self.mlp_decoder(x)  # Shape: (batch_size, 1)

        return output
