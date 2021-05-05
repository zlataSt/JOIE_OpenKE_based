import logging
from joie.base_models import TransE
from joie.base_models import HolE

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

transe = HolE(in_path = "./joie/data/db/insnet/", model_name = "hoie", margin = 1.0, p_norm = 1, dim = 100,
                train_times = 1, alpha = 0.5, use_gpu = False, norm_flag = True)
transe.trainer()
transe.tester()