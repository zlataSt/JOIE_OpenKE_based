import logging

import torch.nn as nn

from joie.openke.config import Trainer, Tester
from joie.openke.data import TrainDataLoader, TestDataLoader
from joie.openke.module.loss import SoftplusLoss, MarginLoss
from joie.openke.module.model import DistMult as Distmult
from joie.openke.module.model import TransE as Transe
from joie.openke.module.model.HolE import HolE as Hole
from joie.openke.module.strategy import NegativeSampling

log = logging.getLogger(__name__)


class BaseModel(nn.Module):

    def __init__(self, in_path, model_name="transe", margin=5.0, p_norm=1, dim=50,
                 train_times=10, alpha=1.0, use_gpu=False, norm_flag=True):
        super().__init__()
        self.in_path = in_path
        self.model_name = model_name
        self.margin = margin,
        self.p_norm = p_norm,
        self.dim = dim,
        self.train_times = train_times,
        self.alpha = alpha,
        self.use_gpu = use_gpu,
        self.norm_flag = norm_flag,
        self.load_train_data(),
        self.define_loss()
        self.opt_method = 'sgd'

    def get_dim(self):
        return int(self.dim[0])

    def get_train_times(self):
        return int(self.train_times[0])

    def get_alpha(self):
        return int(self.alpha[0])

    def get_margin(self):
        return int(self.margin[0])

    def get_p_norm(self):
        return int(self.p_norm[0])

    def get_use_gpu(self):
        return (self.use_gpu[0])

    def get_norm_flag(self):
        return (self.norm_flag[0])

    def log_module_info(self):
        log.info('Initializing %s model with parameters:\n'
                 'file_name: %s \n'
                 'dim: %s \n'
                 'margin: %s \n'
                 'p_norm: %s \n',
                 self.model_name, self.in_path,
                 self.get_dim(),
                 self.get_margin(), self.get_p_norm())

    def load_train_data(self):
        self.train_dataloader = TrainDataLoader(
            in_path=self.in_path,
            nbatches=100,
            threads=8,
            sampling_mode="normal",
            bern_flag=1,
            filter_flag=1,
            neg_ent=25,  # получить по кол-ву сущностей, так как пропорция 1 к 1 В ОТДЕЛЬНОЙ ФУНКЦИИ
            # но это не точно, будем смотреть у китайца
            neg_rel=0)  # тоже либо 1 к 1 либо как у китайца
        return self.train_dataloader

    def define_model(self):
        raise NotImplemented

    def define_loss(self):
        raise NotImplemented

    def neg_sample(self, **kwargs):
        model = NegativeSampling(
            model=self.define_model(),
            loss=self.define_loss(),
            batch_size=self.train_dataloader.get_batch_size(),
            regul_rate=1.0
        )
        self.model = model
        return self.model

    def load_test_data(self):
        self.test_dataloader = TestDataLoader(self.in_path, "link")
        return self.test_dataloader

    def trainer(self):
        # print(self.get_use_gpu())
        trainer = Trainer(model=self.neg_sample(),
                          data_loader=self.load_train_data(),
                          train_times=self.get_train_times(),
                          alpha=self.get_alpha(),
                          use_gpu=self.get_use_gpu(),
                          opt_method=self.opt_method)
        trainer.run()
        model_path = self.in_path + '/checkpoint/' + self.model_name + '.ckpt'
        self.model_a.save_checkpoint(model_path)
        return model_path

    def tester(self):
        model_path = self.trainer()
        self.model_a.load_checkpoint(model_path)
        tester = Tester(model=self.model_a, data_loader=self.load_test_data(), use_gpu=self.get_use_gpu())
        tester.run_link_prediction(type_constrain=False)


class TransE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.neg_sample()
        self.log_module_info()

    def define_model(self):
        transe = Transe(
            ent_tot=self.load_train_data().get_ent_tot(),
            rel_tot=self.load_train_data().get_rel_tot(),
            dim=self.get_dim(),
            p_norm=self.get_p_norm(),
            norm_flag=self.get_norm_flag())
        self.model_a = transe
        return self.model_a

    def define_loss(self):
        transe_loss = MarginLoss(margin=self.get_margin())
        self.loss = transe_loss
        return self.loss


class DistMult(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info()
        self.opt_method = 'adagrad'
        # self.neg_sample(regul_rate = 1.0)

    def define_model(self):
        distmult = Distmult(
            ent_tot=self.load_train_data().get_ent_tot(),
            rel_tot=self.load_train_data().get_rel_tot(),
            dim=self.get_dim())
        self.model_a = distmult
        return self.model_a

    def define_loss(self):
        distmult_loss = SoftplusLoss()
        self.loss = distmult_loss
        return self.loss


class HolE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_module_info()
        self.opt_method = 'adagrad'
        # self.neg_sample(regul_rate = 1.0)

    def define_model(self):
        hole = Hole(
            ent_tot=self.load_train_data().get_ent_tot(),
            rel_tot=self.load_train_data().get_rel_tot(),
            dim=self.get_dim())
        self.model_a = hole
        return self.model_a

    def define_loss(self):
        hole_loss = SoftplusLoss()
        self.loss = hole_loss
        return self.loss
