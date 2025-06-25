import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patch embedding via conv projection.
    Converts spatial map to token sequence.
    """

    def __init__(self, in_ch, embed_dim, kernel_size, stride, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H_out, W_out]
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x, H_out, W_out


class TransformerBlock(nn.Module):
    """
    Lightweight Transformer encoder for token sequences.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, depth=2, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(0, 1)  # [N, B, C]
        x = self.encoder(x)
        x = x.transpose(0, 1)  # [B, N, C]
        return x


class XFeat(nn.Module):
    """
    XFeat backbone with transformer downsampling + conv fusion.
    Output: [B, fusion_dim, H/16, W/16]
    """

    def __init__(
        self,
        pretrained=True,
        top_k=500,
        trans_dim=64,
        fusion_dim=256,
        trans_heads=4,
        fusion_heads=8,
        trans_mlp_ratio=4.0,
        fusion_mlp_ratio=4.0,
        trans_depth=2,
        fusion_depth=1,
        **kwargs
    ):
        super().__init__()
        # 1) Frozen torch.hub backbone
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=pretrained, top_k=top_k
        )
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) Transformer-based downsampling to H/8
        self.patch_embed = PatchEmbed(
            in_ch=3, embed_dim=trans_dim, kernel_size=8, stride=8  # H→H/8
        )
        self.trans_block = TransformerBlock(
            embed_dim=trans_dim,
            num_heads=trans_heads,
            mlp_ratio=trans_mlp_ratio,
            depth=trans_depth,
        )

        # 3) Frozen backbone conv features at H/8
        #    will get from self.model.net(x_prep)

        # 4) Conv fusion & downsample to H/16, expand to fusion_dim
        self.fusion_block = nn.Sequential(
            nn.Conv2d(
                64 + trans_dim,
                fusion_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # H/8→H/16
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            *[
                nn.Conv2d(
                    fusion_dim,
                    fusion_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            ]
            * (fusion_depth)
        )

    def forward(self, x):
        # preprocess
        x_prep, _, _ = self.model.preprocess_tensor(x)
        # transformer path -> new_feat [B, trans_dim, H/8, W/8]
        tokens, H8, W8 = self.patch_embed(x_prep)
        tokens = self.trans_block(tokens)
        B, N, D = tokens.shape  # N=H8*W8
        new_feat = tokens.transpose(1, 2).view(B, D, H8, W8)

        # backbone conv feature [B,64,H/8,W/8]
        M1, _, _ = self.model.net(x_prep)

        # concat -> [B,64+trans_dim,H/8,W/8], then conv fuse to [B,fusion_dim,H/16,W/16]
        fused = torch.cat([M1, new_feat], dim=1)
        out = self.fusion_block(fused)
        return out
