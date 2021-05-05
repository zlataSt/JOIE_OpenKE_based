from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# родительских класс для функций потерь
class Loss(nn.Module, ABC):
    def __init__(self, margin=2.5):
        super().__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        score = self.forward(*args, **kwargs)
        return score.cpu().data.numpy()

# лоссы для TransE, CG, CT
class MarginLoss(Loss):

    def __init__(self, margin=0.5):
        super().__init__(margin=margin)

    def forward(self, p_score, n_score):
        return (
           torch.max(p_score - n_score, -self.margin)
        ).mean() + self.margin

# Лоссы для DistMult, HolE
class SoftPlusLoss(Loss):

    def __init__(self):
        super(SoftPlusLoss, self).__init__()
        self.criterion = nn.Softplus()

    def forward(self, p_score, n_score):
        return (
            self.criterion(-p_score).mean() +
            self.criterion(n_score).mean()
        ) / 2

# Лоссы для внутренней модели (если не нужна иерархичность обнуляем соответсв. параметры
class IntraViewModelLoss(nn.Module):

    def __init__(self,
                 margin_ontonet=2.5,
                 margin_ha=1.0):
        super().__init__()
        self.margin_ontonet = nn.Parameter(
            torch.Tensor([margin_ontonet])
        )
        self.margin_ha = nn.Parameter(
            torch.Tensor([margin_ha])
        )

    def forward(self, instance_loss, ontology_loss, ha_loss):
        return (
            instance_loss +
            self.margin_ontonet * ontology_loss +
            self.margin_ha * ha_loss
        )

# общие лоссы для модели JOIE : J = J_Intra + w*J_Cross
class JoieLoss(Loss):

    def __init__(self, margin=1.0):
        super().__init__(margin=margin)

    def forward(self, cross_view_loss, intra_view_loss):
        return (
            intra_view_loss +
            self.margin * cross_view_loss
        )
