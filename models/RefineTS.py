import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from layers.RefineTS.Embed import DataEmbedding_inverted
from layers.RefineTS.Transformer_EncDec import Encoder, EncoderLayer


def normalize_adjacency(adjacency, clamp=True):
    """Normalize the precomputed channel relation prior."""
    adjacency = torch.Tensor(adjacency)
    inverse_degree = torch.diag(1 / torch.sum(adjacency, dim=0))
    normalized = torch.matmul(inverse_degree, adjacency)
    normalized = torch.matmul(torch.eye(normalized.shape[0], normalized.shape[1]), normalized)
    if clamp:
        normalized = torch.clamp(normalized, max=1.0)
    return normalized


class TemporalLearner(nn.Module):
    """Aggregate in the time domain, filter in the frequency domain, and fuse."""

    def __init__(self, d_series, d_core):
        super().__init__()
        # Keep the registered names and initialization order for reproducibility.
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        self.sparsity_threshold = 0.01
        scale = 0.02
        self.layernorm = nn.LayerNorm(d_series)
        self.w = nn.Parameter(scale * torch.randn(2, d_series))
        self.w1 = nn.Parameter(scale * torch.randn(2, d_series))
        self.rb1 = nn.Parameter(scale * torch.randn(d_series))
        self.ib1 = nn.Parameter(scale * torch.randn(d_series))
        self.rb2 = nn.Parameter(scale * torch.randn(d_series))
        self.ib2 = nn.Parameter(scale * torch.randn(d_series))

        # Retain these unused layers for checkpoint keys and seeded initialization.
        self.fc1_real = nn.Linear(d_series, d_series)
        self.fc1_imag = nn.Linear(d_series, d_series)
        self.fc2_real = nn.Linear(d_series, d_series)
        self.fc2_imag = nn.Linear(d_series, d_series)

    def time_domain_global_aggregation(self, features):
        batch_size, channels, _ = features.shape
        temporal_features = self.gen2(F.gelu(self.gen1(features)))
        probabilities = F.softmax(temporal_features, dim=1)
        if self.training:
            sampling_probabilities = probabilities.permute(0, 2, 1).reshape(-1, channels)
            indices = torch.multinomial(sampling_probabilities, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            global_features = torch.gather(temporal_features, 1, indices)
        else:
            global_features = torch.sum(temporal_features * probabilities, dim=1, keepdim=True)
        return global_features.repeat(1, channels, 1)

    def generate_frequency_filter(self, spectrum):
        # Two elementwise complex affine stages with ReLU between them.
        hidden_real = F.relu(
            torch.einsum('bid,d->bid', spectrum.real, self.w[0])
            - torch.einsum('bid,d->bid', spectrum.imag, self.w[1])
            + self.rb1
        )
        hidden_imag = F.relu(
            torch.einsum('bid,d->bid', spectrum.imag, self.w[0])
            + torch.einsum('bid,d->bid', spectrum.real, self.w[1])
            + self.ib1
        )
        filter_real = (
            torch.einsum('bid,d->bid', hidden_real, self.w1[0])
            - torch.einsum('bid,d->bid', hidden_imag, self.w1[1])
            + self.rb2
        )
        filter_imag = (
            torch.einsum('bid,d->bid', hidden_imag, self.w1[0])
            + torch.einsum('bid,d->bid', hidden_real, self.w1[1])
            + self.ib2
        )
        components = torch.stack([filter_real, filter_imag], dim=-1)
        return torch.view_as_complex(F.softshrink(components, lambd=self.sparsity_threshold))

    def frequency_domain_adaptive_filtering(self, features):
        spectrum = torch.fft.rfft(features, dim=1, norm='ortho')
        frequency_filter = self.generate_frequency_filter(spectrum)
        return torch.fft.irfft(
            spectrum * frequency_filter, n=features.shape[1], dim=1, norm='ortho'
        )

    def forward(self, features, *args, **kwargs):
        global_time_features = self.time_domain_global_aggregation(features)
        frequency_features = self.frequency_domain_adaptive_filtering(features)

        # The trained fusion weights expect frequency features in the first D positions.
        fused_features = torch.cat([frequency_features, global_time_features], dim=-1)
        return self.gen4(F.gelu(self.gen3(fused_features))), None


class ChannelLearner(nn.Module):
    """Apply the channel relation prior to low-rank channel adapters."""

    def __init__(self, configs, embedding_dim):
        super().__init__()
        self.rank = configs.rank
        self.node_dim = configs.node_dim
        self.ch_lora_adapter = nn.Parameter(
            torch.empty(configs.enc_in, embedding_dim, self.rank)
        )
        nn.init.xavier_uniform_(self.ch_lora_adapter)
        self.share_proj = nn.Linear(self.rank, self.node_dim, bias=False)
        self.register_buffer('A', normalize_adjacency(configs.adj))

    def forward(self, embeddings):
        ch_adapter = F.relu(self.share_proj(self.ch_lora_adapter))
        ch_adapter = ch_adapter.permute(1, 2, 0)
        ch_adapter = repeat(ch_adapter, 'D d n -> repeat D d n', repeat=embeddings.shape[0])
        embeddings_by_channel = embeddings.transpose(1, 2)
        ch_adapter = torch.einsum('nm,bthn->bthm', self.A, ch_adapter)
        adapted_features = torch.einsum('bDn,bDdn->bdn', embeddings_by_channel, ch_adapter)
        return torch.cat([embeddings_by_channel, adapted_features], dim=1).transpose(1, 2)


class RefineTS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        embed_dim = configs.d_model - configs.node_dim
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, embed_dim, configs.embed, configs.freq, configs.dropout
        )
        self.use_norm = configs.use_norm
        self.encoder = Encoder(
            [
                EncoderLayer(
                    TemporalLearner(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.channel_learner = ChannelLearner(configs, embed_dim)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        channels = x_enc.shape[2]
        embeddings = self.enc_embedding(x_enc, x_mark_enc)
        ca_features = self.channel_learner(embeddings)
        tf_features, _ = self.encoder(ca_features, attn_mask=None)
        forecast = self.projection(tf_features).permute(0, 2, 1)[:, :, :channels]

        if self.use_norm:
            forecast = forecast * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            forecast = forecast + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return forecast

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        forecast = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return forecast[:, -self.pred_len:, :]


# The experiment runner constructs models through this entry point.
Model = RefineTS
