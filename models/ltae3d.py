import copy
import numpy as np
import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoder


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MultiHeadAttention3D_LTAE(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None):
        # v: [BHW, T, d_in]
        BHW, T, _ = v.shape
        d_k, n_head = self.d_k, self.n_head

        q = torch.stack([self.Q for _ in range(BHW)], dim=1).view(-1, d_k)
        k = self.fc1_k(v).view(BHW, T, n_head, d_k)
        k = k.permute(2, 0, 1, 3).reshape(-1, T, d_k)

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))

        v_split = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * BHW, T, -1
        )

        output, attn = self.attention(q.unsqueeze(1), k, v_split, pad_mask)
        # output: [n_head*BHW, 1, d_k]
        attn = attn.view(n_head, BHW, 1, T).squeeze(2)
        output = output.view(n_head, BHW, -1).transpose(0, 1)  # [BHW, n_head, d//n_head]

        return output, attn  # output shape [BHW, n_head, d//n_head]


class MultiHeadAttention3D_Full(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.q_proj = nn.Linear(d_in, n_head * d_k)
        self.k_proj = nn.Linear(d_in, n_head * d_k)
        self.v_proj = nn.Linear(d_in, n_head * d_k)
        self.out_proj = nn.Linear(n_head * d_k, d_in)
        self.scale = d_k ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pad_mask=None):
        # x: [BHW, T, d_in]
        BHW, T, _ = x.shape
        n_head, d_k = self.n_head, self.d_k

        q = self.q_proj(x).view(BHW, T, n_head, d_k).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(BHW, T, n_head, d_k).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(BHW, T, n_head, d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [BHW, n_head, T, T]
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = self.softmax(scores)  # [BHW, n_head, T, T]
        out = torch.matmul(attn, v)  # [BHW, n_head, T, d_k]

        out = out.permute(0, 2, 1, 3).reshape(BHW, T, n_head * d_k)
        out = self.out_proj(out)

        return out, attn  # out: [BHW, T, d_in], attn: [BHW, n_head, T, T]


class LTAE3D(nn.Module):
    def __init__(
            self,
            in_channels=128,
            n_head=8,
            d_k=16,
            mlp=[128, 64],
            dropout=0.2,
            d_model=128,
            T=1000,
            return_att=False,
            positional_encoding=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mlp_dims = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp_dims[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention = MultiHeadAttention3D_LTAE(n_head, d_k, self.d_model)

        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers = []
        for i in range(len(self.mlp_dims) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]),
                    nn.BatchNorm1d(self.mlp_dims[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        B, C, T, H, W = x.shape
        out = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        # positional encoding
        if self.positional_encoder is not None and batch_positions is not None:
            # print(f"bp shape: {batch_positions.shape}")
            bp = (batch_positions.unsqueeze(-1)
                  .repeat(1, 1, H)
                  .unsqueeze(-1)
                  .repeat(1, 1, 1, W))
            bp = bp.permute(0, 2, 3, 1).reshape(B * H * W, T)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention(out, pad_mask)

        if self.attn_mode == "ltae":
            # LTAE output [B, C', H, W]，repeat T times
            out = out.permute(1, 0, 2).reshape(B * H * W, -1)
            out = self.dropout(self.mlp(out))
            out = self.out_norm(out).view(B, H, W, -1).permute(0, 3, 1, 2)
            out = out.unsqueeze(1).repeat(1, T, 1, 1, 1)
            attn = attn.view(self.n_head, B, H, W, T).permute(1, 0, 4, 2, 3)  #  b x head x t x h x w


        if self.return_att:
            return out, attn
        else:
            return out

class TemporalConvFFN(nn.Module):
    def __init__(self, in_channels, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = in_channels * expansion

        self.conv1 = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(3, 1, 1),  # only convolution in T dimension
            padding=(1, 0, 0),
            groups=1 # specific to input channels if DW conv desired
        )
        self.norm1 = nn.BatchNorm3d(hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout3d(dropout)

        self.conv2 = nn.Conv3d(
            hidden_dim,
            in_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=1 # specific to input channels if DW conv desired
        )
        self.norm2 = nn.BatchNorm3d(in_channels)
        self.drop2 = nn.Dropout3d(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.drop2(out)
        return out + residual


class BottleneckTemporalFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 n_head=8,
                 d_k=16,
                 d_model=128,
                 mlp=[256, 128],
                 ):
        super().__init__()

        self.ltae = LTAE3D(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            return_att=True,
            mlp=mlp
        )

        self.temporal_conv = TemporalConvFFN(in_channels)
        self.fuse_conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x, batch_positions=None):

        # LTAE branch
        out_ltae, attn = self.ltae(x, batch_positions)

        # Temporal Conv branch
        out_conv = self.temporal_conv(x)  # [B, C, T, H, W]
        out_ltae = out_ltae.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        out = torch.cat([out_ltae, out_conv], dim=1)
        out = self.fuse_conv(out)

        return out, attn


class BottleneckSpatialAttention(nn.Module):
    """
    Input BxCxTxHxW series data, compute Multi-head Spatial Attention at each patch at each time-step
    Reshape to BTxNxCpp -> (Attn) -> BxTxHxWxC
    """
    def __init__(self, image_size, patch_size, depth, heads, input_dim, emb_dim=256, dropout=0.1, expand=4):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        if isinstance(image_size, int):
            H, W = image_size, image_size
        else:
            H, W = image_size

        assert H % patch_size == 0 and W % patch_size == 0, \
            "H and W must be divisible by patch_size"

        self.H_patch = H // patch_size
        self.W_patch = W // patch_size
        self.N = self.H_patch * self.W_patch

        # patch embedding: conv -> flatten (B*T, N, C*ps*ps)
        self.patch_embed = nn.Conv2d(
            input_dim,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=emb_dim * expand,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.N, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


        self.proj_out = nn.ConvTranspose2d(
            emb_dim,
            input_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        tokens = self.patch_embed(x)  # (B*T, emb, Hp, Wp)
        B_, D, Hp, Wp = tokens.shape
        # flatten -> (B*T, N, D)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = tokens.transpose(1, 2).view(B * T, D, Hp, Wp)
        out_t = self.proj_out(tokens)  # (B*T, C, H, W)
        out = out_t.reshape(B, C, T, H, W)
        return out


class Bottleneck(nn.Module):
    """
    Fuse Spatial & Temporal feature as final bottleneck output.
    Input:  [B, C, T, H, W]
    Fusion: sum / multiply / cross_attention
    """
    def __init__(self,
                 in_channels=64,
                 # Temporal encoder params
                 heads_temporal=8,
                 d_k=16,
                 d_model=128,
                 mlp=[256, 128],
                 # Spatial encoder params
                 image_size=32,
                 patch_size_spatial=4,
                 depth_spatial=2,
                 heads_spatial=4,
                 emb_dim_spatial=256,
                 dropout=0.1,
                 # Cross attention params
                 heads_cross=4,
                 dropout_cross=0.1
                 ):
        super().__init__()

        self.temporal_encoder = BottleneckTemporalFusion(
            in_channels=in_channels,
            n_head=heads_temporal,
            d_k=d_k,
            d_model=d_model,
            mlp=mlp,
        )

        self.spatial_encoder = BottleneckSpatialAttention(
            image_size=image_size,
            patch_size=patch_size_spatial,
            depth=depth_spatial,
            heads=heads_spatial,
            input_dim=in_channels,
            emb_dim=emb_dim_spatial,
            dropout=dropout
        )


        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,
                                                num_heads=heads_cross,
                                                dropout=dropout_cross,
                                                batch_first=True)
        self.ln_cross = nn.LayerNorm(in_channels)
        self.num_heads_cross = heads_cross


    def forward(self, x, batch_positions=None):
        """
        x: [B, C, T, H, W]
        """
        temporal_out, attn = self.temporal_encoder(x, batch_positions)  # [B,C,T,H,W]
        spatial_out = self.spatial_encoder(x)  # [B,C,T,H,W]


        B, C, T, H, W = temporal_out.shape
        N = H * W

        # flatten -> [B*T, N, C]
        def flatten_features(feat):
            return feat.permute(0, 2, 3, 4, 1).reshape(B*T, N, C)

        q = flatten_features(temporal_out)  # (B*T, N, C)
        kv = flatten_features(spatial_out)

        # cross attention (batch_first=True expects [B, N, C])
        attn_out, attn_weights = self.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        attn_out = self.ln_cross(attn_out + q)
        fused = attn_out.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        attn_weights = attn_weights.view(B, T, self.num_heads_cross, N, N)  # [B, T, n_head, N, N]
        attn_weights = attn_weights.permute(2, 0, 3, 1, 4)  # [n_head, B, N, T, N]
        attn_weights = attn_weights.mean(dim=-1)  # [n_head, B, N, T]
        attn_weights = attn_weights.reshape(self.num_heads_cross, B * N, T)  # [n_head, BHW, T]
        attn_weights = attn_weights.reshape(B, self.num_heads_cross, T, H, W)
        return fused, attn_weights

if __name__ == "__main__":
    x = torch.randn(2, 32, 8, 64, 64)
    model = LTAE3D(in_channels=32)
    bp = torch.arange(0, 8).unsqueeze(0).repeat(2, 1)
    print(bp.shape)
    y = model(x, batch_positions=bp)
    print(y.shape)