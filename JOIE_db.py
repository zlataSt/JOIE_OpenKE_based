import logging

from joie.loss import MarginLoss
from joie.models import CrossViewGrouping
from joie.models import TransE
from joie.trainer import JoieTrainer
from joie.utils import load_data, init_emb

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# загрузчик датасетов
data = load_data(
    instance_file='db_insnet_train.txt',
    ontology_files=['db_ontonet_small_train.txt', 'db_HA_train.txt'],
    instype_file='db_InsType_train.txt',
    path_prefix='joie/data/db',
    gen_corrupted_head=False,
    create_test_files=True
)

# первичная инициализация эмбеддингов для фактов
instance_ent_emb, instance_rel_emb = init_emb(
    data,
    files=['db_insnet_train.txt'],
    dim=300,
    file_type='instance'
)

instype_ent_emb, instype_rel_emb = init_emb(
    data,
    files=['db_InsType_train.txt'],
    dim=50,
    file_type='instype'
)

# первичная инициализация эмбеддингов для онтологий
# ontology_ent_emb, ontology_rel_emb = init_emb(
#     data,
#     files=['db_ontonet_small_train.txt', 'db_HA_train.txt'],
#     dim=50,
#     file_type='ontology'
# )
# print(ontology_ent_emb[0])
ontology_ent_emb, ontology_rel_emb = init_emb(
    data,
    files=['db_ontonet_small_train.txt'],
    dim=50,
    file_type='ontology'
)
ha_ent_emb, ha_rel_emb = init_emb(
    data,
    files=['db_HA_train.txt'],
    dim=50,
    file_type='ontology'
)
###############################################################################
#                             INTRA-VIEW MODEL
###############################################################################
common_params = {
    'p_norm': 1.0,  # евклидова норма (1 или 2) для scoring functions
    'margin': 0.5,
}

model_insnet = TransE(
    file_name='db_insnet_train.txt',
    nbatches=10,
    entity_emb=instance_ent_emb,
    rel_emb=instance_rel_emb,
    loss=MarginLoss,
    d1=200, d2=200,
    **common_params
)

# внутренняя модель векторизации графа онтологий
model_ontonet = TransE(
    file_name='db_ontonet_small_train.txt',
    nbatches=10,
    entity_emb=ontology_ent_emb,
    rel_emb=ontology_rel_emb,
    d1=200, d2=200,
    loss=MarginLoss,
    **common_params
)

# иерархическая модель графа онтологий
model_ha = CrossViewGrouping(
    file_name='db_HA_train.txt',
    nbatches=10,
    entity_emb=ha_ent_emb,
    concept_emb=ha_ent_emb,
    d1=200, d2=200,
    loss=MarginLoss,
    **common_params
)
###############################################################################
#                              CROSS-VIEW MODEL
###############################################################################
model_cvt = CrossViewGrouping(
    file_name='db_InsType_train.txt',
    nbatches=10,
    entity_emb=instype_ent_emb,
    concept_emb=instype_ent_emb,
    loss=MarginLoss,
    d1=200, d2=200,
    **common_params
)
##############################################################################
#                         UNITED JOIE MODEL
##############################################################################
JoieTrainer(
    data=data,
    # intra_view_models=[model_insnet],
    intra_view_models=[model_insnet, model_ontonet, model_ha],
    cross_view_model=model_cvt,
    train_times=120,
    save=True,
    path_prefix='./joie/data/db',
).fit()
