import logging
import torch

log = logging.getLogger(__name__)


class Optimizer:
    def __init__(
            self, model, # инициализация объекта класса (оптимизатора)
            # вида оптимизатора
            learning_rate = 0.1, # скорость обучения
            lr_decay = 0.0, # затухание скорости обучения
            weight_decay = 0.0,
            type = 'Adam'
    ):
        self.params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if 'InsType' in model.model.file_name:
            self.params = filter(
                lambda obj: (
                    obj[1].requires_grad and
                    obj[0] != 'model.concept_emb.weight'
                ),
                model.named_parameters()
            )
            self.params = [obj[1] for obj in self.params]
            # создание объекта с параметрами оптимизатора

        # Оптимизатор Адаптивный Градиентный алгоритм
        if type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(
                self.params,
                lr=learning_rate,
                lr_decay=lr_decay,
                weight_decay=weight_decay
            )

        # Оптимизатор AMSGrad
        elif type == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.params,
                lr=learning_rate,
                weight_decay=weight_decay,
                amsgrad=True # оптимизатор AMSGrad - это улучшенный вариант оптимизатора Адам
                # включаем его проставив True при вызове метода, реализующего оптимизатор Адам
            )
        elif type == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # вывод на экран информации по оптимизации
        log.info('Optimizer for %s: %s, learning rate: %s',
                 model.model.file_name, type, learning_rate)

    # возвращение оптимизатора в исходное состояние
    def backprop(self, loss, grad_clip = 2.0):
        self.optimizer.zero_grad() # приравниваем градиенты к нулю
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.params, grad_clip)
        self.optimizer.step() # производим один шаг оптимизации ЦФ
