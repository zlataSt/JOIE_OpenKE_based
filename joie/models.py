import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class BaseModel(nn.Module):

    def __init__(self,
                 nbatches, file_name,
                 entity_emb, d1, d2, p_norm=2,
                 margin=None, loss=None,
                 concept_emb=None, rel_emb=None):
        super().__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.nbatches = nbatches
        self.p_norm = p_norm
        self.entity_emb = entity_emb
        self.rel_emb = rel_emb
        self.concept_emb = concept_emb
        self.file_name = file_name
        self.d1 = d1
        self.d2 = d2
        self.loss = loss

    def log_module_info(self, model_name):
        log.info('Initializing %s model with parameters:\n'
            'file_name: %s \n'
            'd1: %s \n'
            'd2: %s \n'
            'margin: %s \n'
            'p_norm: %s \n',
            model_name, self.file_name,
            self.d1, self.d2,
            self.margin, self.p_norm)

    def calc_loss(self, h, r, t, mode=None):
        raise NotImplemented

    def forward(self, params):
        batch_h = params['batch_h']
        batch_r = params['batch_r']
        batch_t = params['batch_t']
        mode = params.get('mode', None)
        h = self.entity_emb(batch_h)
        t = self.entity_emb(batch_t)
        r = self.rel_emb(batch_r)

        score = self.calc_loss(h, r, t, mode)
        return score

    def predict(self, params):
        score = self.forward(params)
        return score.cpu().data.numpy()


class TransE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info('TransE')

    def calc_loss(self, h, r, t, mode=None):

        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)

        h = h.view(-1, r.shape[0], h.shape[-1])
        t = t.view(-1, r.shape[0], t.shape[-1])
        r = r.view(-1, r.shape[0], r.shape[-1])

        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        return torch.norm(score, self.p_norm, -1).flatten()


class DistMult(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info('DistMult')

    def calc_loss(self, h, t, r, mode=None):
        if mode == 'head_batch':
            score = h * (r * t)
        else:
            score = (h * r) * t
        score = torch.sum(score, -1).flatten()
        return score

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()


class HolE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info('HolE')

    def _conj(self, tensor):
        zero_shape = (list)(tensor.shape)
        one_shape = (list)(tensor.shape)
        zero_shape[-1] = 1
        one_shape[-1] -= 1
        ze = torch.zeros(size=zero_shape, device=tensor.device)
        on = torch.ones(size=one_shape, device=tensor.device)
        matrix = torch.cat([ze, on], -1)
        matrix = 2 * matrix
        return tensor - matrix * tensor

    def _real(self, tensor):
        dimensions = len(tensor.shape)
        return tensor.narrow(dimensions - 1, 0, 1)

    def _imag(self, tensor):
        dimensions = len(tensor.shape)
        return tensor.narrow(dimensions - 1, 1, 1)

    def _mul(self, real_1, imag_1, real_2, imag_2):
        real = real_1 * real_2 - imag_1 * imag_2
        imag = real_1 * imag_2 + imag_1 * real_2
        return torch.cat([real, imag], -1)

    def ccorr(self, a, b):
        a = self._conj(torch.rfft(a, signal_ndim=1, onesided=False))
        b = torch.rfft(b, signal_ndim=1, onesided=False)
        res = self._mul(self._real(a), self._imag(a), self._real(b),
                        self._imag(b))
        res = torch.ifft(res, signal_ndim=1)
        return self._real(res).flatten(start_dim=-2)

    def calc_loss(self, h, t, r, mode=None):
        score = self.ccorr(h, t) * r
        score = torch.sum(score, -1).flatten()
        return score

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()


class CrossViewTransformation(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info('CrossViewTransformation')

    def tanh(self, entity):
        lin = nn.Linear(self.d1, self.d2)
        return torch.tanh(lin(entity)).squeeze()

    def calc_loss(self, h, r, t, mode=None):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        h = self.tanh(h.unsqueeze(0).float())
        l2_norm = torch.norm(t - h, self.p_norm, -1).flatten()
        return l2_norm

    def forward(self, params):
        batch_h = params['batch_h']
        batch_t = params['batch_t']
        mode = params.get('mode', None)
        h = self.entity_emb(batch_h)
        t = self.concept_emb(batch_t)

        score = self.calc_loss(h, None, t, mode)
        return score


class CrossViewGrouping(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info('CrossViewGrouping')

    def calc_loss(self, h, r, t, mode=None):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        l2_norm = torch.norm(t - h, self.p_norm, -1).flatten()
        return l2_norm

    def forward(self, params):
        batch_h = params['batch_h']
        batch_t = params['batch_t']
        mode = params.get('mode', None)
        h = self.entity_emb(batch_h)
        t = self.concept_emb(batch_t)

        score = self.calc_loss(h, None, t, mode)
        return score