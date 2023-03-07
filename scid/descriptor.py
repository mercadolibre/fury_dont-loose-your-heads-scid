from enum import IntFlag
import pickle as pkl
from itertools import islice

import numpy as np
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from .model import MultiTaskLanguageModel


class InPlaceEmbedder:
    def __init__(self, descr_field, price_field, cat_field, cat2vec, device):
        self.cat_field = cat_field
        self.price_field = price_field
        self.descr_field = descr_field
        self.cat2vec = cat2vec
        self.device = device

    def __call__(self, batch):
        tensors = []
        for session in batch:
            session_tensor = []
            for doc in session:
                item_tensor = doc[self.descr_field] + [doc[self.price_field] / 10] + self.cat2vec[doc[self.cat_field]]
                session_tensor.append(item_tensor)
            tensors.append(torch.tensor(session_tensor))
        res = pad_sequence(tensors, batch_first=True)
        if self.device != 'cpu': res = res.to(self.device)
        return res


class PoolingStrategy(IntFlag):
    LAST = 1
    AVG = 2
    MAX = 4


class MTLDescriptorBase(BaseEstimator, TransformerMixin):
    def __init__(self, mtl_checkpoint_fname, cat2vec_fname, device='cpu', mtl_cls=MultiTaskLanguageModel):
        self.mtl_checkpoint_fname = mtl_checkpoint_fname
        self.cat2vec_fname = cat2vec_fname
        self.device = device
        self.mtl_cls = mtl_cls
        self._model = None

    def __getstate__(self):
        state = vars(self).copy()
        state['_model'] = None
        return state

    def __setstate__(self, state):
        vars(self).update(state)

    def sync_resources(self):
        if self._model is None:
            self._model = self.mtl_cls.load_from_checkpoint(
                self.mtl_checkpoint_fname
            ).to(self.device).eval()
            with open(self.cat2vec_fname, 'rb') as f:
                # TODO: saveit in a list format
                cat2vec = {k: v.tolist() for k, v in pkl.load(f).items()}
            self._model.orig_sku_embedder = self.orig_sku_embedder = self._model.sku_embedder
            self._model.sku_embedder = self._sku_embedder = InPlaceEmbedder(
                descr_field='description_vector', price_field='price_bucket',
                cat_field='category_hash', cat2vec=cat2vec, device=self.device
            )

    def fit(self, X, y=None):
        self.sync_resources()
        return self


class UserNavDescriptor(MTLDescriptorBase):
    def __init__(self, mtl_checkpoint_fname, cat2vec_fname, pooling_strategy: PoolingStrategy,
                 session_key='session', device='cpu', batch_size=4096, verbose=False, mtl_cls=MultiTaskLanguageModel):
        super().__init__(mtl_checkpoint_fname, cat2vec_fname, device, mtl_cls)
        self.session_key = session_key
        self.pooling_strategy = pooling_strategy
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, X):
        self.sync_resources()
        res = []
        X = iter(X)
        while True:
            data = list(islice(X, 0, self.batch_size))
            if not data: break
            res.append(self._minibatch_transform(data))

        return np.concatenate(res, axis=0)

    def _minibatch_transform(self, X):
        sessions = []
        for i, doc in enumerate(X):
            sess = doc[self.session_key]
            # filtro para que ande con sigir, hay que sacarlo de aca
            sess = [e for e in sess if 'event_type' not in e or e['event_type'] != 'pageview']
            assert len(sess) > 0
            sessions.append({'orig_pos': i, 'sess': sess})

        sessions.sort(key=lambda x: -len(x['sess']))
        indices = list(range(len(sessions)))
        indices.sort(key=lambda i: sessions[i]['orig_pos'])
        sessions = [e['sess'] for e in sessions]

        lengths = torch.tensor([len(s) for s in sessions])
        if self.device != 'cpu': lengths = lengths.to(self.device)

        with torch.no_grad():
            hidden, tasks = self._model(sessions, lengths)
            res = self.pool(hidden, lengths)[indices]
            if self.device != 'cpu': res = res.cpu()
            return res.numpy()

    def predict_tasks(self, X):
        self.sync_resources()

        sessions = []
        for i, doc in enumerate(X):
            sess = doc[self.session_key]
            sessions.append({'orig_pos': i, 'sess': sess})

        sessions.sort(key=lambda x: -len(x['sess']))
        indices = list(range(len(sessions)))
        indices.sort(key=lambda i: sessions[i]['orig_pos'])
        sessions = [e['sess'] for e in sessions]

        lengths = torch.tensor([len(s) for s in sessions])
        if self.device != 'cpu': lengths = lengths.to(self.device)

        with torch.no_grad():
            hidden, tasks = self._model(sessions, lengths)
            return {k: v[indices] for k, v in tasks.items()}

    def pool(self, hidden, lengths):
        tensors = []
        if self.pooling_strategy & PoolingStrategy.LAST:
            tensors.append(hidden[range(hidden.shape[0]), lengths - 1])

        if self.pooling_strategy & PoolingStrategy.AVG:
            tensors.append(hidden.sum(dim=1) / lengths[:, None])

        if self.pooling_strategy & PoolingStrategy.MAX:
            # not consider padding for max pooling.
            # TODO check if there's a better way
            hidden[hidden == 0] = -float('Inf')
            tensors.append(hidden.max(dim=1).values)

        return torch.cat(tensors, dim=1)


class CIDDescriptor(MTLDescriptorBase):
    def __init__(self, mtl_checkpoint_fname, cat2vec_fname, item_key,
                 session_key='session', device='cpu', batch_size=4096, verbose=False, mtl_cls=MultiTaskLanguageModel):
        super().__init__(mtl_checkpoint_fname, cat2vec_fname, device, mtl_cls)
        self.session_key = session_key
        self.item_key = item_key
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, X):
        self.sync_resources()
        res = []
        X = iter(X)
        while True:
            data = list(islice(X, 0, self.batch_size))
            if not data: break
            res.append(self._minibatch_transform(data))

        return np.concatenate(res, axis=0)

    def _minibatch_transform(self, X):
        data = []
        for i, doc in enumerate(X):
            sess = doc[self.session_key]
            sess = [e for e in sess if e['event_type'] != 'pageview']
            assert len(sess) > 0
            item = doc[self.item_key]
            data.append({
                'orig_pos': i, 'sess': sess,
                'descr': item['description_vector'],
                'price': item['price_bucket'],
                'cat': item['category_hash']
            })

        data.sort(key=lambda x: -len(x['sess']))
        indices = list(range(len(data)))
        indices.sort(key=lambda i: data[i]['orig_pos'])
        sessions = [e['sess'] for e in data]

        descrs = [e['descr'] for e in data]
        descrs = torch.tensor(descrs)
        if self.device != 'cpu': descrs = descrs.to(self.device)
        descrs = descrs / torch.norm(descrs, p=2, dim=1)[:, None]

        prices = [e['price'] for e in data]
        prices = torch.tensor(prices)
        if self.device != 'cpu': prices = prices.to(self.device)

        cats = [self._model.sku_embedder.cat2vec[e['cat']] for e in data]
        cats = torch.tensor(cats)
        if self.device != 'cpu': cats = cats.to(self.device)
        cats = cats / torch.norm(cats, p=2, dim=1)[:, None]

        lengths = torch.tensor([len(s) for s in sessions])
        if self.device != 'cpu': lengths = lengths.to(self.device)

        with torch.no_grad():
            hidden, tasks = self._model(sessions, lengths)
            for task, result in tasks.items():
                result, _ = pad_packed_sequence(
                    pack_padded_sequence(result, lengths.to('cpu'), batch_first=True), batch_first=True
                )
                tasks[task] = result

            # cat features
            next_sku_cat = tasks['next_sku_cat']
            cats = cats[:, None, :].repeat(1, next_sku_cat.shape[1], 1)
            next_sku_cat = next_sku_cat / torch.clip(torch.norm(next_sku_cat, p=2, dim=2), min=0.00001)[:, :, None]

            features = (cats * next_sku_cat).sum(dim=2)
            next_cat_features = self.pool(features, lengths)

            purch_sku_cat = tasks['purch_sku_cat']
            purch_sku_cat = purch_sku_cat / torch.clip(torch.norm(purch_sku_cat, p=2, dim=2), min=0.00001)[:, :, None]

            features = (cats * purch_sku_cat).sum(dim=2)
            purch_cat_features = self.pool(features, lengths)

            # description features
            next_sku_descr = tasks['next_sku_descr']
            descrs = descrs[:, None, :].repeat(1, next_sku_descr.shape[1], 1)
            next_sku_descr = next_sku_descr / torch.clip(torch.norm(next_sku_descr, p=2, dim=2), min=0.00001)[:, :, None]

            features = (descrs * next_sku_descr).sum(dim=2)
            next_descr_features = self.pool(features, lengths)

            purch_sku_descr = tasks['purch_sku_descr']
            purch_sku_descr = purch_sku_descr / torch.clip(torch.norm(purch_sku_descr, p=2, dim=2), min=0.00001)[:, :, None]

            features = (descrs * purch_sku_descr).sum(dim=2)
            purch_descr_features = self.pool(features, lengths)

            # price features
            next_price_features = self.pool(tasks['next_sku_price'][range(len(data)), :, prices.long() - 1], lengths)
            purch_price_features = self.pool(tasks['purch_sku_price'][range(len(data)), :, prices.long() - 1], lengths)

            res = torch.cat([
                next_descr_features, next_cat_features, next_price_features,
                purch_descr_features, purch_cat_features, purch_price_features
            ], dim=1)[indices]
            if self.device != 'cpu': res = res.cpu()
            return res.numpy()

    def pool(self, features, lengths):
        tensors = [
            features[range(features.shape[0]), lengths - 1][:, None],
            (features.sum(dim=1) / lengths)[:, None]
        ]

        # not consider padding for max pooling.
        # TODO check if there's a better way
        features[features == 0] = -float('Inf')
        tensors.append(features.max(dim=1).values[:, None])

        return torch.cat(tensors, dim=1)


metric_cols = [
    'val_next_sku_descr_prec_at_1',
    'val_purch_sku_descr_prec_at_1',
    'val_next_sku_cat_prec_at_1',
    'val_purch_sku_cat_prec_at_1',
    'val_purch_sku_price_acc',
    'val_next_sku_price_acc',
    'val_will_purch_auc', 'val_this_purch_auc'
]




if __name__ == '__main__': main()
