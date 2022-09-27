import logging

from joie.openke.config import Trainer, Tester
from joie.openke.data import TrainDataLoader, TestDataLoader
from joie.openke.module.loss import MarginLoss
from joie.openke.module.model import TransE
from joie.openke.module.strategy import NegativeSampling

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

train_dataloader = TrainDataLoader(
    in_path="./data/db/insnet/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=0,
    filter_flag=0,
    neg_ent=25,
    neg_rel=0)

# dataloader for test


# define the model
transe = TransE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=300,
    p_norm=1,
    norm_flag=True)

# define the loss function
model = NegativeSampling(
    model=transe,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)
test_dataloader = TestDataLoader("./data/db/insnet/", "link")

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1, alpha=1.0, use_gpu=True)
trainer.run()
transe.save_checkpoint('./data/db/checkpoint/insnet_transe.ckpt')

# test the model
transe.load_checkpoint('./data/db/checkpoint/insnet_transe.ckpt')
tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)
