import timm
import torch.nn as nn


class TeacherViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=True
        )

    def forward(self, x):
        return self.model(x)