import logging
import numpy as np
import torch
from abc import ABC
from tqdm import tqdm

from torch.autograd import Variable

from .negative_sampling import NegativeSamplingModel, NegativeSamplingForCG
from .loss import IntraViewModelLoss, JoieLoss, MarginLoss
from .openke.data import TestDataLoader
from .openke.config import Tester
from .optimizer import Optimizer
from .utils import get_batch_size

log = logging.getLogger(__name__)

# базовый класс Учителя
class BaseTrainer(ABC):
    def __init__(self, *args, **kwargs):
        self.train_times = kwargs.get('train_times', 120)
        self.save = kwargs.get('save', True)
        self.path_prefix = kwargs.get('path_prefix', 'joie/data/yago')

    # оптимизация одного шага
    @staticmethod
    def train_one_step(batch, model, optimizer):
        loss = model({
            'batch_h': Variable(
                torch.from_numpy(np.array(batch['batch_h'])).to(torch.int64)
            ),
            'batch_r': Variable(
                torch.from_numpy(np.array(batch['batch_r'])).to(torch.int64)
            ),
            'batch_t': Variable(
                torch.from_numpy(np.array(batch['batch_t'])).to(torch.int64)
            )
        })
        optimizer.backprop(loss)
        return loss.item()

    # обучение одной эпохи
    def train_one_epoch(self, batches, model, optimizer):
        res = 0.0 # суммарные потери на эпоху
        step = 0 # номер шага
        for triplets in batches:
            # оптимизация (нахождение значения функции потерь)
            loss = self.train_one_step(triplets, model, optimizer)
            res += loss # увеличение размера общих потерь
            step += 1 # обвновление счетчика шага
        # Средние потери за эпоху для модели
        log.info('Average loss for model %s: %s',
                 model.__class__.__name__, res / step)
        return res

    # сохранение результатов обучения модели
    def save_model(self, neg_model):
        # если флаг для сохранения результатов обучения
        if not self.save:
            return
        # выясняем на каком датасете обучалась
        dataset = neg_model.model.file_name.split('_')[0]
        file_type = neg_model.model.file_name.split('_')[1]
        # сохранение последней точки обучения в ckpt-файл
        torch.save(
            neg_model.model.state_dict(),
            f'''{dataset}{file_type}_tt{self.train_times}.ckpt'''
        )
    # тестирование модели
    def test_model(self, neg_model):
        # загрузка тестового датасета
        dataset = neg_model.model.file_name.split('_')[1]
        test_dataloader = TestDataLoader(f'{self.path_prefix}/{dataset}/', 'link')
        # инициализация объекта класса Тестировщик
        tester = Tester(
            model=neg_model.model,
            data_loader=test_dataloader,
            use_gpu=False
        )
        # Если модель содержит в себе множество перекрестных ссылок
        if 'InsType' in dataset:
            tester.run_link_prediction(score_tail=False, type_constrain=False)
        else:
            tester.run_link_prediction(type_constrain=False)

    # подгонка модели (общая), подробнее ниже в каждом классе-наследнике
    def fit(self, neg_model, optimizer, data):
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            batches = data.gen_batches(
                file_name=neg_model.model.file_name,
                nbatches=neg_model.model.nbatches
            )
            log.info('batch_size %s', get_batch_size(data, neg_model.model))
            loss = self.train_one_epoch(batches, neg_model, optimizer)
            log.info('Current loss for %s at %s epoch: %s',
                     neg_model.model.file_name, epoch, loss)

        self.save_model(neg_model)
        self.test_model(neg_model)

# Класс для обучения JOIE-модели
class JoieTrainer(BaseTrainer):
    # конструктор класса тренировщика общей модели JOIE
    def __init__(self, intra_view_models, cross_view_model, data, **kwargs):
        super().__init__(**kwargs)
        # инициализируем:
        self.data = data # обучающие данные
        self.joie_loss = JoieLoss() # потери совокупной JOIE-модели
        self.intra_view_loss = IntraViewModelLoss() # потери внутренней модели JOIE
        # внутренние модели: фактологическая, онтологическая, иерархическая
        self.intra_view_models = intra_view_models
        # перекрестная модель
        self.cross_view_model = cross_view_model
        # внутренние модели с "неправильными" триплетами
        # проход идет по всем внутренним моделям (фактологической, онтологической, иерархической)
        # в итоге получим несколько моделей с "неправильными" триплетами
        self.intra_view_neg_models = [NegativeSamplingModel( # обращаемся к классу
            model=model,
            loss=model.loss(),
            data=data
        ) for model in self.intra_view_models] # проход идет по всем внутренним моделям

        # перекрестная модель с "неправильными" триплетами, на выходе одна модель
        self.cross_view_neg_model = NegativeSamplingModel(
            model=self.cross_view_model,
            loss=self.cross_view_model.loss(),
            data=data
        )

        # инициализация объектов-оптимизаторов для внутренних моделей
        self.intra_view_optimizers = [
            Optimizer(model=self.intra_view_neg_models[0], learning_rate=0.01, type = 'Adagrad'), # фактологической
            Optimizer(model=self.intra_view_neg_models[1], learning_rate=0.01, type = 'Adagrad'), # онтологической
            Optimizer(model=self.intra_view_neg_models[2], learning_rate=0.01, type='Adagrad')  # иерархической
            # последняя для JOIE опциональна, лучше ее вынести
        ]

        # инициализация объекта-оптимизатора для перекрестной модели
        # по статье ищем минимум ЦФ методом стохастического градиентного спуска
        self.cross_view_optimizer = Optimizer(
            self.cross_view_neg_model,
            learning_rate=0.01,
            type = 'Adam'
        )

        # главный (общий с тремя внутренними моделями?) оптимизатор для фактологической внутренней модели
        # по статье ищем минимум ЦФ методом стохастического градиентного спуска
        self.main_intra_view_optimizer = Optimizer(
            model=self.intra_view_neg_models[0],
            learning_rate=0.01,
            type = 'Adam'
        )

        # общий JOIE-оптимизатор
        self.joie_optimizer = Optimizer(
            self.intra_view_neg_models[0],
            learning_rate=0.01,
            type='Adam'
        )

    # обучение одной эпохи для внутренней (Intra-View) модели
    def train_one_intra_view_epoch(self, data):
        # список потерь по 2-3 рассматриваемым внутренним моделям
        cur_loss = []
        # проход по всем внутренним негативным моделям и внутренним оптимизаторам
        for model, optimizer in zip(self.intra_view_neg_models, self.intra_view_optimizers):
            # создание пакетов с "правильными" и "неправильными" триплетами для данной модели
            batches = data.gen_batches(
                file_name=model.model.file_name,
                nbatches=model.model.nbatches
            )
            # логирование информации по размеру полученного батча и обучению одной эпохи
            log.info('%s: batch_size %s',
                     model.model.__class__.__name__,
                     get_batch_size(data, model.model))
            _loss = self.train_one_epoch(
                batches, model, optimizer
            )

            # добавление потерь данной модели в список
            cur_loss.append(_loss)

            # Логирование текущих потерь
            log.info('Current loss for %s: %s',
                     model.model.file_name, _loss)

        # общие потери внутренней модели с учетом потерь данной модели
        main_loss = self.intra_view_loss(*cur_loss)
        # финальная оптимизация внутренней модели (по трем лоссам)
        # вопрос - на tensorflow тоже оптимизируют покусочно или целиком
        self.main_intra_view_optimizer.backprop(loss=main_loss)
        # логирование текущих потерь по внутренней модели
        log.info('Current loss for IntraView model: %s', main_loss)
        return main_loss

    # обучение одной эпохи для прекрестной модели
    def train_one_cross_view_epoch(self, data):
        # создание пакетов с "правильными" и "неправильными" триплетами для данной модели
        batches = data.gen_batches(
            file_name=self.cross_view_model.file_name,
            nbatches=self.cross_view_model.nbatches
        )

        # логирование информации по размеру полученного батча и обучению одной эпохи
        log.info('%s: batch_size %s',
                 self.cross_view_model.__class__.__name__,
                 get_batch_size(data, self.cross_view_model))
        _loss = self.train_one_epoch(
            batches, self.cross_view_neg_model, self.cross_view_optimizer
        )
        # логирование текущих потерь по внутренней модели
        log.info('Current loss for %s: %s',
                 self.cross_view_model.file_name, _loss)
        return _loss

    # подгонка параметров по общей модели JOIE
    def fit(self, *args):
        # сколько раз обучаем модель
        training_range = tqdm(range(self.train_times))
        # проход по всем эпохам
        for epoch in training_range:
            # потери внутренней модели за одну эпоху
            intra_view_loss = self.train_one_intra_view_epoch(self.data)
            # потери перекрестной модели за одну эпоху
            cross_view_loss = self.train_one_cross_view_epoch(self.data)
            # оптимизация общей функции потерь JOIE
            # по результатам, полученным внутренней и перекрестной моделью
            loss = self.joie_loss(intra_view_loss, cross_view_loss)
            self.joie_optimizer.backprop(loss=loss)
            # логирование величины текущей функции потерь для данной эпохи
            log.info('Current loss for Joie model at %s epoch: %s',
                     epoch, loss)

        # для всех моделей, имеющих "неправильные" триплеты
        for model in [*self.intra_view_neg_models, self.cross_view_neg_model]:
            self.save_model(model) # сохранение параметров модели (результатов обучения)
            self.test_model(model) # сохранение результатов тестирования модели

class UnitTrainer(BaseTrainer):
    def __init__(self, unit_models, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        # функция потерь для TransE
        # лучше реализовать выбор в зависимости от названия модели
        # и желательно чтобы в классе-модели был конструктор,
        # ссылающийся на класс потери данного типа
        # либо связать эти два класса иначе
        self.unit_loss = MarginLoss()
        # модель подается в виде списка из одного элемента
        # мб лучше просто ссылкой на объект
        self.unit_models = unit_models
        # инициализируем объект класса моделей с "неправильным" триплетом
        # в данном случае можно без цикла, если модель бы передавалась не в вивде списка
        self.unit_neg_models = [NegativeSamplingModel(
            model=model,
            loss=model.loss(),
            data=data
        ) for model in self.unit_models]

        # Оптимизатор для данной модели
        self.unit_optimizers = [
            Optimizer(model=self.unit_neg_models[0], learning_rate=1.0),
        ]

    def train_one_base_epoch(self, data):
        # инициализация модели, модели с "неправильными" триплетами и оптимизатора
        m = self.unit_models[0]
        m_neg = self.unit_neg_models[0]
        m_opt = self.unit_optimizers[0]
        # генерируем батч на эпоху
        batches = data.gen_batches(
            file_name=m.file_name,
            nbatches=m.nbatches
        )
        # логирование информации о батчах, потерях и обучение одной эпохи
        log.info('%s: batch_size %s',
                 m.__class__.__name__,
                 get_batch_size(data, m))
        _loss = self.train_one_epoch(
            batches, m_neg, m_opt
        )
        log.info('Current loss for %s: %s',
                 m.file_name, _loss)

        return _loss

    # подгонка параметров
    def fit(self, *args):
        # инициализация модели и данных
        m = self.unit_neg_models[0]
        d = self.data
        # сколько раз обучаем модель
        training_range = tqdm(range(self.train_times))
        # на каждой эпохе
        for epoch in training_range:
            # потери базовой модели в одной эпохе
            unit_loss = self.train_one_base_epoch(d)
            # откат к исходному состоянию оптимизатора,
            # с которым все валится
            self.unit_optimizers[0].backprop(loss=unit_loss)
            # логирование
            log.info('Current loss for Base model at %s epoch: %s',
                     epoch, unit_loss)
        self.save_model(m)
        self.test_model(m)
