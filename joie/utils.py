import numpy as np

import torch
import torch.nn as nn

from joie.loader import LoadData

# функция, стартующая загрузку данных из файла
def load_data(
        instance_file: str, # тип объекта, по которому получаем данные по фактам
        ontology_files: list, # тип объекта, по которому получаем данные по онтологиям
        instype_file: str, # тип объекта, по которому получаем данные по перекрестным ссылкам
        gen_corrupted_head: bool = False, # флаг того, что генерируем "голову"
        create_test_files: bool = False, # флаг генерации файлов с тестовыми наборами данных
        path_prefix: str = 'joie/data/yago' # корневая папка с файлами
):
    # непосредственно загрузка данных
    # инициализация объетка класса Загрузчика
    data = LoadData(
        instance_file, ontology_files, instype_file,
        gen_corrupted_head=gen_corrupted_head,
        path_prefix=path_prefix
    )
    # чтение переданных файлов
    data.read_all_files()
    # создание файлов с идентификаторами элементов графа
    data.create_id_files()
    # если нужно генерировать файлы с тестовой выборкой
    if create_test_files:
        # создаем файлы с идентификаторами сущностей и отношений
        data.create_test_id_files()
    return data


def uniform_on_surface(ndim, entities_size):
    vec = np.random.randn(ndim, entities_size)
    vec /= np.linalg.norm(vec, axis=0)
    return torch.from_numpy(vec.T).float()

# функция для случайного выбора значений для инициализации эмбеддингов
# распределения, из которых выбираются элементы отличаются
# в зависимости от выбранного типа инициализации
def _weights(weight, dim, init_type='xavier'):

    emb = nn.Embedding(weight, dim)
    if init_type == 'xavier':
        # заполнение тензора значениями по методу Ксавье
        nn.init.xavier_uniform_(emb.weight.data)
    elif init_type == 'uniform':
        # заполнение тензора значениями из равномерного распределения
        emb.weight.data = uniform_on_surface(dim, weight)
    #print('Weight embedding: ', emb)
    return emb


# первичная инициализация эмбеддингов
def init_emb(data, files, dim, file_type):

    # списки для эмбеддингов отношений и сущностей
    ent_emb, rel_emb, con_emb = [], [], []
    init_type = 'xavier' # первичная инициализация методом Xavier
    # подбор нужного файла
    if (file_type == 'instance') or (file_type == 'instype'):
        file = files[0]
    else: # первичная инициализацяи из равномерного распределения
        init_type = 'uniform'
        file = 'ontology_files' # инициализация сущностей-онтологий
    _ent_emb = _weights(
        data.get_entities_size(file), dim, init_type=init_type
    )
    _rel_emb = _weights(
        data.get_rel_size(file), dim, init_type=init_type
    )
    # _con_emb = _weights(
    #     data.get_concept_size(file), dim, init_type=init_type
    # )
    #print('file: ', file)
    ent_emb.append(_ent_emb.weight.data)
    rel_emb.append(_rel_emb.weight.data)
    #con_emb.append(_con_emb.weight.data)

    ent_weight = torch.cat(ent_emb, dim=0)
    rel_weight = torch.cat(rel_emb, dim=0)
    #con_weight = torch.cat(con_emb, dim=0)

    ent_emb = nn.Embedding(data.get_entities_size(file), dim)
    rel_emb = nn.Embedding(data.get_rel_size(file), dim)
    #con_emb = nn.Embedding(data.get_concept_size(file), dim)

    ent_emb.weight.data.copy_(ent_weight)
    rel_emb.weight.data.copy_(rel_weight)
    #print('ent emb: ', ent_emb)
    #print('rel emb: ', rel_emb)
    #con_emb.weight.data.copy_(con_weight)

    return ent_emb, rel_emb

# получение словаря, где ключ - id слова, а значение - само слово
def get_id2term_dict(file_name):
    return {value: key for key, value in get_id_dict(file_name).items()}

# получение словаря, где ключ - id слова, а значение - само слово из файла
# внутри -  get_id2term_dict
def get_id_dict(file_name):
    ent_dict = dict()
    with open(file_name, 'r') as file:
        for row in file:
            if '\t' not in row:
                continue
            row = row.rstrip()
            row = row.split('\t')
            ent_dict[row[0]] = int(row[1])
    return ent_dict


# функция для получения размера батча
def get_batch_size(data, model):
    return data.get_triplets_size(model.file_name) // model.nbatches
