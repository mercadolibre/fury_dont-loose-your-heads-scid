import mmh3
import numpy as np
import torch
from memoized_property import memoized_property
from sklearn.metrics import roc_auc_score

from .nn import FastNN
from .timeit import timeit


class EmbIndex:
    def __init__(self, vectors, id2index, device, exact):
        self.device = device
        self.vectors = vectors
        self.id2index = id2index
        self.exact = exact
        if not exact:
            self.nn = FastNN().fit(vectors)

    @memoized_property
    def vectors_normed(self):
        res = torch.tensor(self.vectors).to(self.device)
        res /= torch.norm(res, p=2, dim=1)[:, None]
        return res

    @classmethod
    def build(cls, vectors, device, exact):
        embs_dict = {}

        with timeit('make emb dict'):
            for v in vectors:
                if v.sum() == 0: continue
                k = cls.vector2id(v)
                if k in embs_dict:
                    # Sanity check, duplicate vectors have the same hash
                    assert np.all(embs_dict[k] == v)
                else:
                    embs_dict[k] = v

        ids, vectors = zip(*sorted(embs_dict.items()))
        vectors = np.asarray(vectors)
        id2index = {id: index for index, id in enumerate(ids)}

        return cls(vectors, id2index, device, exact)

    def prec_at_k_delete(self, input, target, k):
        input = input / torch.norm(input, p=2, dim=1)[:, None]

        target_indices = self.get_indices(target)
        if input.device.type != self.device:
            input = input.to(self.device)
        top_indices = torch.topk(input @ self.vectors_normed.t(), k=k, dim=1).indices

        positive = [0] * k
        for i in range(k):
            positive[i] += (top_indices[:, i] == target_indices).sum()
        return positive, len(target_indices)

    def prec_at_k_accum(self, k):
        return PrecAtKAccumulator(self, k, self.device)

    def to(self, device):
        self.vectors = torch.tensor(self.vectors).to(device)
        return self

    @classmethod
    def vector2id(cls, vector):
        assert isinstance(vector, np.ndarray) or isinstance(vector, list)
        return mmh3.hash(vector) + hash(tuple(vector))

    def get_indices(self, matrix: torch.Tensor, skip_nones=False):
        assert isinstance(matrix, torch.Tensor)
        if matrix.device.type != 'cpu': matrix = matrix.to('cpu')

        # can't do this with torch
        matrix = matrix.numpy().copy()
        ids = []
        included = []
        for i, vector in enumerate(matrix):
            vector_id = self.vector2id(vector)
            if skip_nones and vector_id not in self.id2index:
                continue
            elif skip_nones:
                included.append(i)
            ids.append(self.id2index[self.vector2id(vector)])
        res = torch.tensor(ids)
        if self.device != 'cpu': res = res.to(self.device)
        if skip_nones:
            return res, included
        else:
            return res


class PrecAtKAccumulator:
    def __init__(self, index, k, device):
        self.k = k
        self.index = index
        self.total = 0
        self.positive = torch.zeros(k, device=device)
        self.device = device

    def add_stats(self, input: torch.Tensor, target: torch.Tensor, skip_nones=False):
        input = input / torch.norm(input, p=2, dim=1)[:, None]

        if input.device.type != self.device:
            input = input.to(self.device)

        if skip_nones:
            target_indices, included = self.index.get_indices(target, skip_nones=skip_nones)
            input = input[included]
        else:
            target_indices = self.index.get_indices(target)

        if self.index.exact:
            top_indices = torch.topk(input @ self.index.vectors_normed.t(), k=self.k, dim=1).indices
        else:
            top_indices = torch.tensor(self.index.nn.query(input.numpy(), num_neigbours=self.k).astype(int))

        for i in range(self.k):
            self.positive[i] += (top_indices[:, i] == target_indices).sum()
        self.total += len(target_indices)

    def get_metrics(self):
        prec = np.cumsum(self.positive.to('cpu').numpy()) / self.total
        res = {f'prec_at_{i + 1}': val for i, val in enumerate(prec)}
        res['n'] = float(self.total)
        return res

    def log_metrics(self, model, task_name):
        for name, val in self.get_metrics().items():
            model.log(f'{task_name}_{name}', val, prog_bar=False)


class PriceMetricsAccumulator:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.error = 0
        self.mape = 0

    def add_stats(self, input: torch.Tensor, target: torch.Tensor):
        if input.shape[0] == 0: return

        if input.shape[1] == 10:
            input = torch.argmax(input, dim=1) + 1
        else:
            input = torch.clip(input.view(-1).round(), min=1, max=9)
        target = target.view(-1)
        self.error += ((input - target) ** 2).sum()
        self.mape += (torch.abs(input - target) / target.clip(min=0.1)).sum()
        self.total += len(target)
        self.correct += (input == target).sum()

    def get_metrics(self):
        return dict(acc=self.correct / self.total,
                    rmse=self.error ** 0.5 / self.total,
                    mape=self.mape / self.total,
                    n=float(self.total))

    def log_metrics(self, model, task_name):
        for name, val in self.get_metrics().items():
            model.log(f'{task_name}_{name}', val, prog_bar=False)


class RocAUCAccumulator:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add_stats(self, input: torch.Tensor, target: torch.Tensor):
        self.y_true.extend(target.view(-1).cpu().numpy().tolist())
        self.y_pred.extend(input.view(-1).cpu().numpy().tolist())

    def log_metrics(self, model, task_name):
        for name, val in self.get_metrics().items():
            model.log(f'{task_name}_{name}', val, prog_bar=False)

    def get_metrics(self):
        return dict(auc=roc_auc_score(self.y_true, self.y_pred), n=float(len(self.y_pred)))
