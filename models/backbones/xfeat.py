import torch
import torch.nn as nn

class XFeat(nn.Module):
    """
    XFeat backbone loaded from torch.hub, fused with learnable downsampling and channel/spatial expansion
    Args:
        model_name (str): The name of the xfeat model variant (ignored)
        pretrained (bool): Whether to load pretrained weights
        top_k (int): Number of top features (ignored here)
    """

    def __init__(self, model_name="XFeat", pretrained=True, top_k=500, **kwargs):
        super().__init__()
        # --- 1) Frozen backbone ---
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat",
            pretrained=pretrained, top_k=top_k
        )
        for param in self.model.parameters():
            param.requires_grad = False

        # --- 2) Additional downsampling path (H→H/8) ---
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),  # H→H/2
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), # H/2→H/4
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), # H/4→H/8
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # --- 3) Fusion & expansion block (channels 128→256, spatial downsample H/8→H/16) ---
        self.fusion_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # H/8→H/16
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),  # refine
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 1) Preprocess input
        x_prep, _, _ = self.model.preprocess_tensor(x)

        # 2) Frozen backbone first features [B,64,H/8,W/8]
        M1, _, _ = self.model.net(x_prep)

        # 3) New learnable features [B,64,H/8,W/8]
        new_feat = self.block1(x_prep)

        # 4) Concat along channel dim → [B,128,H/8,W/8]
        fused = torch.cat([M1, new_feat], dim=1)

        # 5) Expand channels & downsample → [B,256,H/16,W/16]
        out = self.fusion_block(fused)
        return out
