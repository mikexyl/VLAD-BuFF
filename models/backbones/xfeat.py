import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patch embedding using a Conv2d projection + LayerNorm, flattening to token sequence.
    Output shape: tokens [B, N, C], and spatial dims H, W for reshaping.
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
        x = self.proj(x)  # → [B,embed_dim,H',W']
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # → [B, N=H'*W', embed_dim]
        x = self.norm(x)
        return x, H_out, W_out


class TransformerBlock(nn.Module):
    """
    Simple Transformer encoder stack for token sequences.
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x shape: [B, N, C]
        # PyTorch Transformer expects [N, B, C]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        return x


class XFeat(nn.Module):
    """
    XFeat backbone fused with transformer-based downsampling and expansion.
    """

    def __init__(
        self,
        pretrained=True,
        top_k=500,
        model_name="XFeat",
        embed_dim1=64,
        embed_dim2=256,
        depth1=2,
        depth2=2,
        heads1=4,
        heads2=8,
        mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__()
        # Frozen torch.hub backbone
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=pretrained, top_k=top_k
        )
        for p in self.model.parameters():
            p.requires_grad = False

        # Transformer-based downsampling to H/8, channels=embed_dim1
        # conv proj = stride=8 patch embed
        self.patch_embed1 = PatchEmbed(3, embed_dim1, kernel_size=8, stride=8)
        self.trans_block1 = TransformerBlock(
            embed_dim=embed_dim1, num_heads=heads1, mlp_ratio=mlp_ratio, depth=depth1
        )

        # Fusion & expansion: input channels sum=64+embed_dim1
        # project+downsample by 2 to get H/16, channels=embed_dim2
        self.patch_embed2 = PatchEmbed(
            64 + embed_dim1, embed_dim2, kernel_size=2, stride=2
        )
        self.trans_block2 = TransformerBlock(
            embed_dim=embed_dim2, num_heads=heads2, mlp_ratio=mlp_ratio, depth=depth2
        )

    def forward(self, x):
        # preprocess
        x_prep, _, _ = self.model.preprocess_tensor(x)
        # backbone features [B,64,H/8,W/8]
        M1, _, _ = self.model.net(x_prep)

        # transformer downsampling path
        tokens1, H1, W1 = self.patch_embed1(x_prep)  # → [B, N1, embed_dim1]
        tokens1 = self.trans_block1(tokens1)
        B, N1, C1 = tokens1.shape
        feat1 = tokens1.transpose(1, 2).view(B, C1, H1, W1)

        # concatenate fuse
        fused = torch.cat([M1, feat1], dim=1)  # [B, 64+embed_dim1, H/8, W/8]

        # transformer expansion & downsample
        tokens2, H2, W2 = self.patch_embed2(fused)  # → [B, N2, embed_dim2]
        tokens2 = self.trans_block2(tokens2)
        _, N2, C2 = tokens2.shape
        out = tokens2.transpose(1, 2).view(B, C2, H2, W2)
        return out
