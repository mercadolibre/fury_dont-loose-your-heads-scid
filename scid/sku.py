import numpy as np
import torch
import pickle as pkl
from .metrics import EmbIndex


class AttributeEmbedding:
    """
    Levanta las matrices correspondiente a los atributos de un sku, de forma tal que podemos
    usar al attributo de un sku_id, accediendo a la fila correspondiente
    """

    def __init__(self, fname, matrices):
        self.fname = fname
        self.matrices = matrices
        self.loaded = False
        self.data = None
        
    def load_data(self):
        if self.loaded: return

        if self.fname.endswith('npz'):
            data = np.load(self.fname)
        elif self.fname.endswith('pkl'):
            with open(self.fname, 'rb') as f:
                data = pkl.load(f)
        else:
            raise RuntimeError(f'invalid extension for {self.fname}')
        
        data = {k: data[k] for k in self.matrices}
        for k in self.matrices:
            data[k] = torch.tensor(data[k]).float()
        
        self.data = data
        self.loaded = True
        return self

    def __getstate__(self):
        state = vars(self).copy()
        state['data'] = None
        state['loaded'] = False
        return state

    def __setstate__(self, state):
        vars(self).update(state)

    def to(self, device):
        self.load_data()
        self.data = {k: v.to(device) for k, v in self.data.items()}
        return self
    
    def emb(self, attr_name, batch):
        return self.data[attr_name][batch]


class FrozenEmbedder:
    """
    Abstraccion py torch style que dado un batch con sku_id calcula los embeddings concatenando 
    descripcion, precio y categoria
    """
    attributes = 'descr', 'price', 'cat'
    
    def __init__(self, attr_embs):
        self.attr_embs = attr_embs
        self._sku_emb = None

    def __getstate__(self):
        state = vars(self).copy()
        state['_sku_emb'] = None
        return state

    def __setstate__(self, state):
        vars(self).update(state)
    @property
    def sku_emb(self):
        if self._sku_emb is None:
            self.attr_embs.load_data()
            data = []
            for k in self.attributes:
                data.append(self.attr_embs.data[k] / (10 if k == 'price' else 1))
            self._sku_emb = torch.cat(data, dim=1)
        return self._sku_emb

    def to(self, device):
        self.attr_embs = self.attr_embs.to(device)
        self._sku_emb = self.sku_emb.to(device)
        return self
    
    def __call__(self, batch):
        return self.sku_emb[batch]

    def get_prec_index(self, attr, device, exact):
        assert attr in self.attributes

        vectors = self.attr_embs.data[attr].to('cpu').numpy().copy()[1:]
        return EmbIndex.build(vectors, device=device, exact=exact)


class MeliFrozenEmbedder(FrozenEmbedder):
    attributes = 'title', 'domain'

    def __call__(self, batch):
        batch, prices = batch
        return torch.cat([self.sku_emb[batch], prices[:, :, None]], dim=2)
