import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ltae3d import  Bottleneck


def pad_temporal_dim(x, n_stages):
    T = x.shape[2]
    factor = 2 ** (n_stages - 1)
    remainder = T % factor
    if remainder != 0:
        pad_len = factor - remainder
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))  # pad order: (W_left,W_right,H_left,H_right,T_left,T_right)
    else:
        pad_len = 0
    return x, pad_len

def pad_batch_positions(bp, n_stages):
    T = bp.shape[1]
    factor = 2 ** (n_stages - 1)
    remainder = T % factor
    if remainder != 0:
        pad_len = factor - remainder
        last_val = bp[:, -1:].expand(-1, pad_len)
        bp = torch.cat([bp, last_val], dim=1)
    else:
        pad_len = 0
    return bp, pad_len


class ST3DUNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        # temporal params
        image_size=1024,
        encoder_widths=[32, 32, 32, 64 ],
        decoder_widths=[16, 16, 32, 64],
        mlp=[128, 64],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        n_head=16,
        d_model=128,
        d_k=4,
        # spatial params
        patch_size_spatial=16,
        depth_spatial=4,
        heads_spatial=4,
        emb_dim_spatial=128,
        dropout=0.1,

        # bottleneck fusion params
        heads_cross=4,
        dropout_cross=0.1

    ):

        super().__init__()
        self.n_stages = len(encoder_widths)

        # encoder
        self.in_conv = ConvBlock3D([input_dim, encoder_widths[0], encoder_widths[0]])
        self.down_blocks = nn.ModuleList(
            DownConvBlock3D(encoder_widths[i], encoder_widths[i+1], k=str_conv_k, s=str_conv_s, p=str_conv_p)
            for i in range(self.n_stages - 1)
        )

        # temporal attention at bottleneck
        self.temporal_attention = Bottleneck(
            in_channels=encoder_widths[-1],
            heads_temporal=n_head,
            d_k=d_k,
            d_model=d_model,
            mlp=mlp,
            image_size=image_size//2**(len(encoder_widths)-1),
            patch_size_spatial=patch_size_spatial,
            depth_spatial=depth_spatial,
            heads_spatial=heads_spatial,
            emb_dim_spatial=emb_dim_spatial,
            dropout=dropout,
            heads_cross=heads_cross,
            dropout_cross=dropout_cross
        )

        self.temporal_aggregator = Temporal_Aggregator_3D()

        self.up_blocks = nn.ModuleList(
            UpConvBlock3D(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i-1],
                d_skip=encoder_widths[i-1],
                k=str_conv_k, s=str_conv_s, p=str_conv_p
            )
            for i in range(self.n_stages-1, 0, -1)
        )

        self.out_conv = nn.Sequential(
            nn.Conv3d(decoder_widths[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, output_dim, kernel_size=1)
        )

    def forward(self, x, batch_positions=None):
        x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        x, pad_len = pad_temporal_dim(x, self.n_stages)
        batch_positions, _ = pad_batch_positions(batch_positions, self.n_stages)
        feature_map = []
        out = self.in_conv(x)
        feature_map.append(out)

        for down in self.down_blocks:
            out = down(out)
            feature_map.append(out)
            if batch_positions is not None:
                batch_positions = batch_positions[:, ::2]

        out, attn = self.temporal_attention(out, batch_positions)
        for i, up in enumerate(self.up_blocks):
            skip = feature_map[-(i+2)]
            skip = skip.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
            skip = self.temporal_aggregator(skip, attn_mask=attn)  #[B, C[-i+2], H // ?, W // ?]

            out = up(out, skip)

        out = self.out_conv(out)
        if pad_len > 0:
            out = out[:, :, :-pad_len, :, :]
        out = out.permute(0, 2, 1, 3, 4)  # B, T, num_class, H, W
        return out


class Temporal_Aggregator_3D(nn.Module):
    def __init__(self):
        super(Temporal_Aggregator_3D, self).__init__()

    def forward(self, x, pad_mask=None, attn_mask=None):

        B, T, C, H, W = x.shape

        # [B, n_head, T, H_b, W_b]
        B_, n_head, Tb, Hb, Wb = attn_mask.shape
        attn = attn_mask

        # ============ Spatial interpolation to the spatial resolution of skip ============
        if (Hb, Wb) != (H, W):
            attn = attn.reshape(B_ * n_head, Tb, Hb, Wb)
            attn = F.interpolate(attn, size=(H, W),
                                 mode="bilinear", align_corners=False)
            attn = attn.view(B, n_head, Tb, H, W)

        # ============ The length of sequence interpolated to skip ============
        if Tb != T:
            # 当前 attn: [B, n_head, Tb, H, W]
            attn = attn.permute(0, 1, 3, 4, 2)  # [B, n_head, H, W, Tb]
            attn = attn.reshape(B * n_head * H * W, 1, Tb)  # [BNHW, 1, Tb]
            attn = F.interpolate(attn, size=T, mode='linear', align_corners=False)
            attn = attn.view(B, n_head, H, W, T).permute(0, 1, 4, 2, 3)

        groups = torch.stack(x.chunk(n_head, dim=2))  # [n_head, B, T, C//n_head, H, W]
        groups = groups.permute(1, 0, 2, 3, 4, 5)  # [B, n_head, T, C//n_head, H, W]

        out = attn[:, :, :, None, :, :] * groups
        weighted = out.permute(1, 0, 2, 3, 4, 5)  #  [n_head, B, T, C//n_head, H, W]
        weighted = torch.cat([g for g in weighted], dim=2)  # B, T, C, H, W
        weighted = weighted.permute(0, 2, 1, 3, 4) # B, C, T, H, W
        return weighted




# 3D conv blocks
class ConvBlock3D(nn.Module):
    def __init__(self, nkernels, norm='batch'):
        super().__init__()
        layers = []
        for i in range(len(nkernels)-1):
            layers.append(
                nn.Conv3d(nkernels[i], nkernels[i+1], 3, padding=1)
            )
            layers.append(nn.BatchNorm3d(nkernels[i+1]))
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownConvBlock3D(nn.Module):
    def __init__(self, d_in, d_out, k=3, s=2, p=1):
        super().__init__()
        self.down = nn.Conv3d(d_in, d_in, kernel_size=k, stride=s, padding=p)
        self.conv1 = ConvBlock3D([d_in, d_out])
        self.conv2 = ConvBlock3D([d_out, d_out])

    def forward(self, x):
        x = self.down(x)
        x = self.conv1(x)
        return x + self.conv2(x)


class UpConvBlock3D(nn.Module):
    def __init__(self, d_in, d_out, d_skip, k=3, s=2, p=1):
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv3d(d_skip, d_skip, 1),
            nn.BatchNorm3d(d_skip),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose3d(d_in, d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm3d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvBlock3D([d_out + d_skip, d_out])
        self.conv2 = ConvBlock3D([d_out, d_out])

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_conv(skip)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return x + self.conv2(x)


if __name__ == "__main__":
    x = torch.randn(2, 8, 4, 1024, 1024)  # [B, T, C, H, W]
    bp = torch.arange(1, 8).unsqueeze(0).repeat(2, 1)
    model = ST3DUNet(input_dim=4, output_dim=6, encoder_widths=[32, 32, 32, 64], decoder_widths=[16, 32, 32, 64])
    y = model(x, bp)
    print(y.shape)  # expected: [2, 6, 32, 64, 64]
