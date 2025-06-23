import torch
import torch.nn as nn

class XFeat(nn.Module):
    """
    XFeat backbone loaded from torch.hub
    Args:
        model_name (str): The name of the xfeat model variant (ignored, kept for compatibility)
        pretrained (bool): Whether to use pretrained weights
        top_k (int): Number of top features to keep
    """
    def __init__(self, model_name="XFeat", pretrained=True, top_k=500, **kwargs):
        super().__init__()
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=pretrained, top_k=top_k)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x, _, _ = self.model.preprocess_tensor(x)
        M1, _, _ = self.model.net(x)
        M1 = torch.nn.functional.normalize(M1, dim=1)
        return M1