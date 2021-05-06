import logging
import numpy as np
from collections import defaultdict
from random import randint, random, shuffle


log = logging.getLogger(__name__)


class LoadData:
    def __init__(
            self,
            instance_file, ontology_files, instype_file,
            path_prefix,
            gen_corrupted_head=True,
            gen_corrupted_tail=False
    ):
        self.path_prefix = path_prefix # путь к директории с файлами
        self.entity_dict = {} # словарь для хранения сущностей-фактов
        self.rel_dict = {} # словарь для хранения связей (отношений)
        self.concept_dict = {} # словарь для хранения сущностей-понятий

        self.entities_size = 0 # количество сущностей-фактов
        self.rel_size = 0 # количество сущностей отношений
        self.concept_size = 0 # количество сущностей онтологий

        self.instance_file = instance_file # файл с графом фактов
        self.ontology_files = ontology_files # файл с графом онтологий
        self.instype_file = instype_file # файл с графом перекрестных ссылок

        self.triplets = defaultdict(list) # объект словарного типа для хранения триплетов
        # в отличие от словаря не выдает ошибку если пустой
        self.all_triplets = 0 # количество триплетов
        self.gen_corrupted_head = gen_corrupted_head # меняем ли "голову" триплета на случайную
        self.gen_corrupted_tail = gen_corrupted_tail # меняем ли "хвост" триплета на случайный

        self._file_info = defaultdict(dict) # информация о файлах

        self._batch_iter = None

    # чтение триплета из файла
    def prepare_triplet(self, triplet, file):
        # работа со строкой
        triplet = triplet.rstrip()
        triplet = triplet.split('\t')

        h = triplet[0] # "входящая" сущность
        r = triplet[1]# сущность-связь
        t = triplet[2] # "входящая" сущность

        # отдельная обработка для файла с перекрсетными ссылками
        # if 'InsType' in file:
        #     # если "головы" нет в словаре сущностей
        #     if h not in self.entity_dict:
        #         # получаем идентификатор для нее
        #         self.entity_dict[h] = self.get_ent_dict(self.instance_file)[h]
        #         # обновляем счетчик кол-ва сущностей
        #         self.entities_size += 1
        #         # если "головы" нет в словаре сущностей
        #     if t not in self.entity_dict:
        #         # получаем идентификатор для него
        #         self.entity_dict[t] = self.get_ent_dict('ontology_files')[t]
        #         # обновляем счетчик кол-ва сущностей
        #         self.entities_size += 1
        #
        #     # получаем триплет для отображения перекрестной ссылки
        #     # кортеж вида (идентификатор сущности-факта, идентификатор сущности-понятия)
        #     triplet2id = (
        #         self.get_ent_dict(self.instance_file)[h],
        #         0,
        #         self.get_ent_dict('ontology_files')[t]
        #     )
        #     # добавляем полученный триплет в список триплетов по ключу-имени файла
        #     self.triplets[file].append(triplet2id)
        #     return triplet2id

        # для файлов, не содержащих перекрестные ссылки
        # если "головы" нет в словаре сущностей
        if h not in self.entity_dict:
            # получаем идентификатор для нее
            self.entity_dict[h] = self.entities_size
            # обновляем счетчик кол-ва сущностей
            self.entities_size += 1

        if r not in self.rel_dict:
            self.rel_dict[r] = self.rel_size
            self.rel_size += 1
        # если "хвоста" нет в словаре сущностей
        if t not in self.entity_dict:
            # получаем идентификатор для него
            self.entity_dict[t] = self.entities_size
            # обновляем счетчик кол-ва сущностей
            self.entities_size += 1
        # получаем триплет в виде кортежа
        triplet2id = (
            self.entity_dict[h],
            self.rel_dict[r],
            self.entity_dict[t]
        )
        # добавляем полученный триплет в список триплетов по ключу-имени файла
        self.triplets[file].append(triplet2id)
        return triplet2id

    @staticmethod
    # первичная инициализация векторов для эмбеддингов нулями
    def init_vector(size):
        return np.zeros(shape=(size,))

    # меняем выбранную сущность на случайную из графа
    ############## нужно добавить проверку всего триплета на "неправильность"################
    ############## и желательно делать замену с tph и hpt, чтобы учесть типы связей (1-to-N, N-to-1, ...) ####
    ############## как и делается генерация в коде автора JOIE ####################

    def neg_entity(self, entity, file):
        while True:
            # случайным образом выбираем сущность по ее месту в файле
            corrupted = randint(0, self.get_entities_size(file) - 1)
            # проверка на несовпдание случайно выбранной сущности с заменяемой
            if corrupted != entity:
                break
        return corrupted

    # чтение "сырого" файла
    def _read_file(self, file):
        log.info('Load of %s started', file) # логирование того какой файл грузим
        # построчное чтение открываемого файла + создания триплета из данных в строке
        with open(f'{self.path_prefix}/{file}', 'r', encoding = 'utf8') as f:
            i = 0
            for row in f:
                i += 1
                yield self.prepare_triplet(row, file)

            # информация о триплетах из файлов без перекрестных ссылок
            #if 'InsType' not in file:
            self._file_info[file] = {
                'all_triplets': i,
                'entities_size': len(self.entity_dict),
                'rel_size': len(self.rel_dict),
                'ent_dict': self.entity_dict,
                'rel_dict': self.rel_dict
            }
            # else:
            #     # информация о триплетах из файлов с перекрестными ссылками
            #     self._file_info[file] = {
            #         'all_triplets': i,
            #         'entities_size': len(self.entity_dict),
            #         'rel_size': len(self.rel_dict),
            #         'concept_size': len(self.concept_dict),
            #         'ent_dict': self.entity_dict,
            #         'concept_dict': self.concept_dict,
            #         'rel_dict': self.rel_dict
            #     }
            # логирование информации о кол-ве триплетов,
            # сущностей и связей в обучающей выборке
            log.info('Train sample for %s model contains: '
                     '%s triplets, %s entities, %s relations',
                     file, i, self.entities_size, self.rel_size)
        # логирование завершения загрузки файла
        log.info('Load of %s ended', file)

    # обнуление параметров триплетов (первичная инициализация)
    def nullify_params(self):
        self.entity_dict = {}
        self.rel_dict = {}
        self.concept_dict = {}
        self.entities_size = 0
        self.rel_size = 0
        self.concept_size = 0

    # считываем все файлы из папки
    def read_all_files(self):
        self.nullify_params() # ставим начальные параметры для множества триплетов
        list(self._read_file(self.instance_file))

        self.nullify_params()

        # чтение файлов с онтологиями и выставление
        # параметров по триплетам для каждого
        for file in self.ontology_files:
            list(self._read_file(file))
        self._file_info['ontology_files'] = {
            'all_triplets': sum(len(self.triplets[file])
                                for file in self.ontology_files),
            'entities_size': len(self.entity_dict),
            'rel_size': len(self.rel_dict),
            'ent_dict': self.entity_dict,
            'rel_dict': self.rel_dict
        }

        # чтение файла с перекрсетными ссылками
        self.nullify_params()
        list(self._read_file(self.instype_file))

    # загрузка батча (пакета данных)
    def load_batch(self, file, nbatches):
        # размер пакета
        batch_size = self.get_triplets_size(file) // nbatches
        batch_seq_size = batch_size*2
        # инициализация матрицы "входящих" сущностей
        h = self.init_vector(batch_seq_size)
        # инициализация матрицы связей
        r = self.init_vector(batch_seq_size)
        # инициализация матрицы "исходящих" сущностей
        t = self.init_vector(batch_seq_size)

        # работа с триплетами из батча
        for i in range(batch_size):
            triplet = next(self._batch_iter)

            h[i] = triplet[0]
            r[i] = triplet[1]
            t[i] = triplet[2]

            # случайная величина
            random_val = random()
            # # # "неправильные" триплеты (negative, corrupted triplets)
            # # # по идее оба флага замены "головы" и "хвоста" не могут быть True
            # # # так как мы меняем либо одно, либо другое
            if self.gen_corrupted_head or self.gen_corrupted_tail:
                # вероятность того что сменим сущность каждого типа = 0.5
                if random_val < 0.5:
                    h[i + batch_size] = self.neg_entity(h[i], file)
                else:
                    t[i + batch_size] = self.neg_entity(t[i], file)
            #меняем "голову"
            elif self.gen_corrupted_head:
                h[i + batch_size] = self.neg_entity(h[i], file)
                t[i + batch_size] = t[i]
            # меняем "хвост"
            elif self.gen_corrupted_tail:
                h[i + batch_size] = h[i]
                #создаем "неправильный" триплет для перекрестных ссылок
                #можем сменить только одну сущность
                if 'InsType' in file:
                    t[i + batch_size] = self.neg_entity(t[i], 'ontology_files')
            r[i + batch_size] = r[i]
        return {
            'batch_h': h,
            'batch_r': r,
            'batch_t': t
        }

    # создание пакетов для обучения и проверки моделей
    def gen_batches(self, file_name, nbatches):
        self._batch_iter = iter(self.triplets[file_name])
        # генерируем пакеты по их количеству, которое выставляем при запуске обучения
        for i in range(nbatches):
            try:
                yield self.load_batch(file_name, nbatches)
            except StopIteration:
                self._batch_iter = iter(self.triplets[file_name])
                yield self.load_batch(file_name, nbatches)

    # запись в файл сущностей, связей и триплетов
    # в формате "название элемента   id согласно порядковому номеру"
    def _write_triplet_size(self, file):
        _dataset = file.split('_')[1]
        # сущности
        with open(f'{self.path_prefix}/{_dataset}/entity2id.txt', 'w', encoding = 'utf8') as f:
            f.write(f'{self.get_entities_size(file)}\n')
        # связи
        with open(f'{self.path_prefix}/{_dataset}/relation2id.txt', 'w', encoding = 'utf8') as f:
            f.write(f'{self.get_rel_size(file)}\n')
        # триплеты
        with open(f'{self.path_prefix}/{_dataset}/train2id.txt', 'w', encoding = 'utf8') as f:
            f.write(f'{self.get_triplets_size(file)}\n')

    # создание файлов, представляющих триплеты в виде идентификаторов
    # для обучения модели
    def create_id_files(self):
        files = [self.instance_file, *self.ontology_files, self.instype_file]
        for file in files:
            self._write_triplet_size(file)
            dataset = file.split('_')[1]
            heads, rels, tails = set(), set(), set()
            with open(f'{self.path_prefix}/{dataset}/train2id.txt', 'a+') as f:
                for head, rel, tail in self.triplets[file]:
                    heads.add(head)
                    rels.add(rel)
                    tails.add(tail)
                    f.write(f'{head}\t{tail}\t{rel}\n')

            with open(f'{self.path_prefix}/{dataset}/entity2id.txt', 'a+', encoding = 'utf8') as f:
                for key, value in self.get_ent_dict(file).items():
                    if value in heads or value in tails:
                        f.write(f'{key}\t{value}\n')
                # if 'InsType' in file:
                #     for key, value in self.get_concept_dict(file).items():
                #         if value in heads or value in tails:
                #             f.write(f'{key}\t{value}\n')

            with open(f'{self.path_prefix}/{dataset}/relation2id.txt', 'a+', encoding = 'utf8') as f:
                for key, value in self.get_rel_dict(file).items():
                    if value in rels:
                        f.write(f'{key}\t{value}\n')

    # создание файлов, представляющих триплеты в виде идентификаторов
    # для тестирования модели
    def create_test_id_files(self):
        valid_size = 0 # размер валидационной  выборки
        test_size = 0 # размер тестовой выборки
        # список файлов
        files =  [self.instance_file, *self.ontology_files, self.instype_file]
        # проход по всем файлам
        for file in files:
            file_test = file.replace('train', 'test')
            _dataset = file.split('_')[1]
            _triplets = []
            # читаем тестовую выборку из файла в "сыром" виде
            with open(f'{self.path_prefix}/{file_test}', 'r', encoding = 'utf8') as f:
                for row in f: # построчно
                    triplet = row.rstrip()
                    triplet = triplet.split('\t')
                    # if 'InsType' in file: # для файла с перекрестными ссылками
                    #     _triplets.append((
                    #         self.get_ent_dict(self.instance_file)[triplet[0]],
                    #         0,
                    #         self.get_ent_dict('ontology_files')[triplet[2]]
                    #     ))
                    #else:
                        # для файла с онтологиями
                    h = triplet[0]
                    #print('Test h:', h)
                    # "входящая" сущность
                    r = triplet[1]  # сущность-связь
                    #print('Test r:', r)
                    t = triplet[2]
                    #print('Test t:', t)
                    if h not in self.entity_dict:
                        # получаем идентификатор для нее
                        self.entity_dict[h] = self.entities_size
                        # обновляем счетчик кол-ва сущностей
                        self.entities_size += 1
                    if r not in self.rel_dict:
                        self.rel_dict[r] = self.rel_size
                        self.rel_size += 1
                    # если "хвоста" нет в словаре сущностей
                    if t not in self.entity_dict:
                        # получаем идентификатор для него
                        self.entity_dict[t] = self.entities_size
                        # обновляем счетчик кол-ва сущностей
                        self.entities_size += 1
            with open(f'{self.path_prefix}/{file_test}', 'r', encoding='utf8') as f:
                for row in f:  # построчно
                    triplet = row.rstrip()
                    triplet = triplet.split('\t')
                    if file in self.ontology_files:
                        file = 'ontology_files'
                    _triplets.append((
                        self.get_ent_dict(file)[triplet[0]],
                        self.get_rel_dict(file)[triplet[1]],
                        self.get_ent_dict(file)[triplet[2]]
                    ))
            # размер проверочной выборки
            _valid_size = int(len(_triplets) * 0.05) + 1
            valid_size += _valid_size
            # перемешивание списка триплетов
            shuffle(_triplets)
            # данные для тестирования модели
            test_data = _triplets[_valid_size:]
            test_size += len(test_data)
            # данные для валидации модели
            valid_data = _triplets[:_valid_size]

            # файл для тестирования с индексами элементов триплетов
            with open(f'{self.path_prefix}/{_dataset}/test2id.txt', 'w', encoding = 'utf8') as f:
                f.write(f'{len(test_data)}\n')
                for head, rel, tail in test_data:
                    border = self.get_entities_size(file)
                    if _dataset == 'InsType' and head > border-1:
                        f.write(f'{head-border}\t{tail}\t{rel}\n')
                    else:
                        f.write(f'{head}\t{tail}\t{rel}\n')

            # файл для валидации с индексами элементов триплетов
            with open(f'{self.path_prefix}/{_dataset}/valid2id.txt', 'w', encoding = 'utf8') as f:
                f.write(f'{len(valid_data)}\n')
                for head, rel, tail in valid_data:
                    border = self.get_entities_size(file)
                    if _dataset == 'InsType' and head > border-1:
                        f.write(f'{head-border}\t{tail}\t{rel}\n')
                    else:
                        f.write(f'{head}\t{tail}\t{rel}\n')

    # функция для получения количества сущностей
    def get_entities_size(self, file_name):
        return self._file_info[file_name]['entities_size']

    # функция для получения количества связей
    def get_rel_size(self, file_name):
        return self._file_info[file_name]['rel_size']

    # функция для вывода количества триплетов
    def get_triplets_size(self, file_name):
        return self._file_info[file_name]['all_triplets']

    # функция для получения файла-словаря сущностей
    def get_ent_dict(self, file_name):
        return self._file_info[file_name]['ent_dict']

    # функция для получения файла-словаря отношений
    def get_rel_dict(self, file_name):
        return self._file_info[file_name]['rel_dict']

    # функция для получения файла-словаря концептов
    def get_concept_dict(self, file_name):
        return self._file_info[file_name]['concept_dict']
