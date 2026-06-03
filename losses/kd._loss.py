import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):

    def __init__(self,
                 temperature=4.0,
                 alpha=0.5):

        super().__init__()

        self.temperature = temperature
        self.alpha = alpha

    def forward(
            self,
            student_logits,
            teacher_logits,
            labels):

        ce_loss = F.cross_entropy(
            student_logits,
            labels
        )

        kd_loss = F.kl_div(
            F.log_softmax(
                student_logits /
                self.temperature,
                dim=1
            ),
            F.softmax(
                teacher_logits /
                self.temperature,
                dim=1
            ),
            reduction='batchmean'
        )

        loss = (
            self.alpha * ce_loss
            +
            (1 - self.alpha)
            * kd_loss
        )

        return loss