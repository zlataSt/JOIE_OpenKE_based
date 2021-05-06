import logging
import torch.nn as nn

from joie.utils import get_batch_size

log = logging.getLogger(__name__)


# лучше переименовать в NegativeSamplingModel, так как данный класс используется для любой
# входящей в JOIE модели - как базовой, так и перекрестной
class NegativeSamplingModel(nn.Module):
    def __init__(
            self, model, loss, data
    ):
        # инициализация объекта класса
        super(NegativeSamplingModel, self).__init__()
        self.data = data # датасет
        self.model = model # модель
        self.loss = loss # функция потерь
        #self.regul_rate = regul_rate
        self.batch_size = get_batch_size(data, model) # размер обучающего пакета (батча)

    # получение значения scoring function для "правильных" триплетов
    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size] # срез по "правильным" триплетам
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    # получение значения scoring function для "неправильных" триплетов
    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:] # срез по "неправильным" триплетам
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    # получение двух матриц с триплетами двух типов
    # для вычисления лосс-функции в дальнейшем
    def forward(self, data):
        # функция достоверности
        score = self.model(data)
        # для "правильных триплетов"
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.loss(p_score, n_score)
        # if self.regul_rate != 0:
        #     loss_res += self.regul_rate * self.model.regularization(data)
        return loss_res

# можно удалить, так как функция потерь для модели Cross Grouping
# вычисляется на основе расстояния между сущностью и концептом

class NegativeSamplingForCG(nn.Module):
    def __init__(self, model=None, loss=None, data=None):
        super(NegativeSamplingForCG, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = get_batch_size(data, model)

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def forward(self, data):
        score = self.model(data)
        p_score = self._get_positive_score(score)
        n_score = 0.0
        loss_res = self.loss(p_score, n_score)
        # if self.regul_rate != 0:
        #     loss_res += self.regul_rate * self.model.regularization(data)
        return loss_res
