import numpy as np
from collections import defaultdict

import torch
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def ensure_not_float64(tensor):
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


class BatchCollator:
    """
    Dado un batch de sesiones genera lo necesario para entrenar el modelo
    input: [[{sku_id: <number>, product_action: <str>}]]

    output: 
      - seq_lengths: size: batch_size.  tensor con la longitud de las sesiones
      - batch: size (batch_size, max_seq_len). Tiene los sku_ids de cada sesion
      - tasks: diccionario {task_name: tensor}, donde tensor tiene size (batch_size, max_seq_len).
      - masks: diccionario {task_name: tensor}, donde tensor tiene size (batch_size, max_seq_len) y dice si la tarea task_name esta definida en cada evento
    """

    sku_id_field = 'sku_id'
    task_names = ['will_purch', 'this_purch', 'next_sku', 'purch_sku']

    def is_purch(self, evt):
        return evt['product_action'] != 'detail'

    def get_tasks(self, sess):
        is_purch_fn = self.is_purch
        sku_id_field = self.sku_id_field

        is_purch = is_purch_fn(sess[-1])
        tasks = []
        for i, doc in enumerate(sess):
            doc_tasks = dict(will_purch=float(is_purch))
            if i < len(sess) - 1:
                doc_tasks['next_sku'] = sess[i + 1][sku_id_field]
            if is_purch:
                doc_tasks['purch_sku'] = sess[-1][sku_id_field]
            doc_tasks['this_purch'] = float(is_purch_fn(doc))
            tasks.append(doc_tasks)
        return tasks

    def collate(self, batch):

        masks = defaultdict(list)
        tasks = defaultdict(list)
        for sess in batch:
            sess_tasks = defaultdict(list)
            sess_masks = defaultdict(list)

            for event_tasks in self.get_tasks(sess):
                for task_name in self.task_names:
                    try:
                        sess_tasks[task_name].append(event_tasks[task_name])
                        sess_masks[task_name].append(True)
                    except KeyError:
                        # Si task_name not esta en event_tasks, quiere decir que la tarea no esta definida para este evento
                        # Por lo que pongo un 0 en su mascara
                        # TODO: fix this
                        sess_tasks[task_name].append(0.0 if 'price' in task_name else 0)
                        sess_masks[task_name].append(False)

            for task_name, task_values in sess_tasks.items():
                tasks[task_name].append(task_values)
                masks[task_name].append(sess_masks[task_name])

        batch, perm_idx, seq_lengths = self.make_batch(batch)

        for task_name in tasks:
            tasks[task_name] = pad_sequence([ensure_not_float64(torch.tensor(tasks[task_name][i])) for i in perm_idx],
                                            batch_first=True)
            masks[task_name] = pad_sequence([torch.tensor(masks[task_name][i]) for i in perm_idx],
                                            batch_first=True)

        tasks = dict(tasks)
        masks = dict(masks)
        return seq_lengths, batch, tasks, masks

    def make_batch(self, batch):
        sku_id_field = self.sku_id_field
        batch = [[e[sku_id_field] for e in s] for s in batch]
        seq_lengths = LongTensor(list(map(len, batch)))
        # sort to be able to pack sequence after
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        batch = pad_sequence([torch.tensor(batch[i]) for i in perm_idx], batch_first=True)
        return batch, perm_idx, seq_lengths

