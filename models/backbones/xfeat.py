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
            batch_first=True,  # (B, N, C)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):  # x: (B, N, C)
        return self.encoder(x)


class XFeat(nn.Module):
    """
    XFeat backbone with optional ViT-style head. If ``return_tokens`` is enabled
    a CLS token is used to build a **global descriptor** which is finally
    projected to ``global_dim`` (default 256).
    """

    def __init__(
        self,
        pretrained=True,
        top_k=500,
        fusion_dim=128,
        trans_heads=2,
        fusion_heads=8,
        trans_mlp_ratio=4.0,
        fusion_mlp_ratio=4.0,
        trans_depth=2,
        fusion_depth=1,
        use_vit_img_head=False,
        return_token=False,
        **kwargs,
    ):
        super().__init__()
        self.use_vit_img_head = use_vit_img_head
        self.return_tokens = return_token
        self.global_dim = fusion_dim
        self.trans_dim = fusion_dim

        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=pretrained, top_k=top_k
        )
        for p in self.model.parameters():
            p.requires_grad = False

        if self.use_vit_img_head:
            self.patch_embed = PatchEmbed(
                in_ch=3, embed_dim=self.trans_dim, kernel_size=8, stride=8
            )
            self.trans_block = TransformerBlock(
                embed_dim=self.trans_dim,
                num_heads=trans_heads,
                mlp_ratio=trans_mlp_ratio,
                depth=trans_depth,
            )

            conv_layers = [
                nn.Conv2d(
                    64 + self.trans_dim,
                    fusion_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            ]
            for _ in range(fusion_depth):
                conv_layers += [
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
            self.fusion_block = nn.Sequential(*conv_layers)
        else:
            # add a block to upsample from 64 to global dim
            self.global_upsample = nn.Sequential(
                nn.Conv2d(64, self.global_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.global_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.global_dim,
                    self.global_dim,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.global_dim),
                nn.ReLU(inplace=True),
            )

        if self.return_tokens:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.token_block = TransformerBlock(
                embed_dim=self.trans_dim,
                num_heads=trans_heads,
                mlp_ratio=trans_mlp_ratio,
                depth=1,
            )
            self.global_proj = nn.Linear(self.trans_dim, self.global_dim)
            nn.init.trunc_normal_(self.cls_token, std=0.02)



    def forward(self, x):
        x_prep, _, _ = self.model.preprocess_tensor(x)

        if self.use_vit_img_head:
            tokens, H8, W8 = self.patch_embed(x_prep)
            tokens = self.trans_block(tokens)

            B, N, D = tokens.shape
            new_feat = tokens.transpose(1, 2).view(B, D, H8, W8)

            M1, _, _ = self.model.net(x_prep)

            fused = torch.cat([M1, new_feat], dim=1)
            feat_map = self.fusion_block(fused)

            if self.return_tokens:
                cls = self.cls_token.expand(B, -1, -1)
                tok_seq = torch.cat([cls, tokens], dim=1)
                tok_seq = self.token_block(tok_seq)
                global_desc = self.global_proj(tok_seq[:, 0])
                return feat_map, global_desc
            return feat_map

        M1, _, _ = self.model.net(x_prep)
        M1 = self.global_upsample(M1)  # [B, 64, H8, W8]

        if self.return_tokens:
            B, C, H8, W8 = M1.shape
            tokens = M1.flatten(2).transpose(1, 2)  # [B, N, C=64]
            cls = self.cls_token.expand(B, -1, -1)
            tok_seq = torch.cat([cls, tokens], dim=1)
            tok_seq = self.token_block(tok_seq)
            global_desc = self.global_proj(tok_seq[:, 0])
            return M1, global_desc

        return M1


# =============================================================================
# Quick shape sanity‑check ----------------------------------------------------
# =============================================================================


def _demo(use_vit=True, return_token=True, H=256, W=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XFeat(pretrained=False, use_vit_img_head=use_vit, return_token=return_token)

    # print the number of all parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}"
    )

    model.to(device)
    model.eval()

    x = torch.randn(2, 3, H, W).to(device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, tuple):
        feat_map, global_desc = out
        print(
            f"feat_map:     {feat_map.shape}  ← expected (2, 256, {H//16}, {W//16})"
            if use_vit
            else f"feat_map:     {feat_map.shape}  ← expected (2, 64, {H//8}, {W//8})"
        )
        print(f"global_desc:  {global_desc.shape} ← expected (2, 256)")
    else:
        print(f"feat_map_only: {out.shape}  ← expected (2, 64, {H//8}, {W//8})")


def main():
    print("\n=== ViT head with tokens ===")
    _demo(use_vit=True, return_token=True)

    print("\n=== ViT head, no tokens ===")
    _demo(use_vit=True, return_token=False)

    print("\n=== Conv‑only path ===")
    _demo(use_vit=False, return_token=False)

    print("\n=== Conv‑only path with token ===")
    _demo(use_vit=False, return_token=True)


if __name__ == "__main__":
    main()
