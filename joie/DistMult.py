import logging

from joie.openke.config import Trainer, Tester
from joie.openke.data import TrainDataLoader, TestDataLoader
from joie.openke.module.loss import SoftplusLoss
from joie.openke.module.model.DistMult import DistMult
from joie.openke.module.strategy import NegativeSampling

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

train_dataloader = TrainDataLoader(
    in_path="./data/db/insnet/",
    nbatches=10,
    threads=8,
    sampling_mode="normal",
    bern_flag=0,
    filter_flag=0,
    neg_ent=2,
    neg_rel=0)

# dataloader for test
test_dataloader = TestDataLoader("./data/db/insnet/", "link")

# define the model
distmult = DistMult(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200)

# define the loss function
model = NegativeSampling(
    model=distmult,
    loss=SoftplusLoss(),
    batch_size=train_dataloader.get_batch_size(),
    regul_rate=1.0
)

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader, train_times=120, alpha=0.01,
                  use_gpu=True, opt_method="adagrad")
trainer.run()
distmult.save_checkpoint('./data/db/checkpoint/insnet_distmult.ckpt')

# test the model
distmult.load_checkpoint('./data/db/checkpoint/insnet_distmult.ckpt')
tester = Tester(model=distmult, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)
