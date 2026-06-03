import timm
import torch.nn as nn


class StudentViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True
        )

    def forward(self, x):
        return self.model(x)