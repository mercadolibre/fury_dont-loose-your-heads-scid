from collections import Counter, defaultdict
from datetime import datetime
from itertools import islice
from random import Random
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from memoized_property import memoized_property
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import GRU, Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from core import fs
from core.models.mlp import LitProgressBar
from core.serialization import iter_jl
from core.settings import ETL_PATH
from .data import SessionDataset, BatchCollator
from .metrics import PriceMetricsAccumulator, RocAUCAccumulator
from .settings import sigir_data_dir
from .sku import FrozenEmbedder, AttributeEmbedding
from ..time_it import timeit


class MultiTaskLoss:
    def __init__(self, enabled_tasks, attr_emb, special_tasks, log_var=None, price_classification=True):
        self.log_var = log_var
        if enabled_tasks:
            enabled_tasks = set(enabled_tasks)
        self.enabled_tasks = enabled_tasks
        self.special_tasks = special_tasks

        self.attr_emb = attr_emb
        self.price_classification = price_classification

    def price_loss(self, input, target, mask):
        if self.price_classification:
            return F.cross_entropy(
                input[mask].view(-1, 10), target[mask].view(-1).long() - 1
            ) / 10

        else:
            mask = mask.view(-1)
            return F.mse_loss(
                input.view(-1)[mask],
                target.view(-1)[mask]
            ) #/ 10

    def __call__(self, outputs, tasks, masks):
        loss = 0

        task_losses = {}
        task_number = 0
        for task, target in tasks.items():
            if task in self.special_tasks:
                if self.enabled_tasks and task not in self.enabled_tasks: continue
                if 'price' in task:
                    task_losses[task] = self.price_loss(outputs[task], target, masks[task])
                else:
                    task_losses[task] = torch.binary_cross_entropy_with_logits(outputs[task].view(-1),
                                                                               target.view(-1)).mean()
                    if task == 'this_purch': task_losses[task] = task_losses[task] * 2

                if self.log_var is not None:
                    loss += task_losses[task] * torch.exp(-self.log_var[task_number]) + self.log_var[task_number]
                else:
                    loss += task_losses[task]

                task_number += 1
            else:
                for attr in self.attr_emb.matrices:
                    if self.enabled_tasks and f'{task}_{attr}' not in self.enabled_tasks: continue
                    pred = outputs[f'{task}_{attr}']
                    bs, max_len, size = pred.shape
                    target_val = self.attr_emb.emb(attr, target)
                    if attr == 'price':
                        task_losses[f'{task}_{attr}'] = self.price_loss(pred, target_val, masks[task])
                    else:
                        m = masks[task].view(bs * max_len)
                        cos_loss = 1 - torch.cosine_similarity(
                            pred.view(bs * max_len, -1)[m],
                            target_val.view(bs * max_len, -1)[m]
                        )
#                         cos_loss = torch.pow(cos_loss, 2)
                        task_losses[f'{task}_{attr}'] = cos_loss.mean() / (2 if attr == 'descr' else 1)

                        # perform softmax with a fixed last layer
                    #                         activation = pred.view(bs * max_len, -1)[m] @ self.attr_emb.data[attr][1:].t()
                    #                         task_losses[f'{task}_{attr}'] = F.cross_entropy(activation, target.view(-1)[m].long() - 1)

                    if self.log_var is not None:
                        loss += (
                            task_losses[f'{task}_{attr}'] * torch.exp(-self.log_var[task_number]) +
                            self.log_var[task_number]
                        )
                    else:
                        loss += task_losses[f'{task}_{attr}']
                    task_number += 1

        return loss, task_losses


class MultiTaskLanguageModel(LightningModule):
    price_classification = True
    descr_size = 50
    price_size = 1
    cat_size = 50
    sku_embedding_size = descr_size + price_size + cat_size

    task_families = 'next_sku', 'purch_sku'
    attributes = 'descr', 'cat', 'price'

    # TODO: remove the device from the constructor!
    # It makes trouble when loading a checkpoint into cpu
    def __init__(self, sku_embedder_fname, device, max_epochs, data_size=None,
                 batch_size=64, learning_rate=3e-4,
                 item_adapter_depth=2,
                 gru_input_size=1.0, gru_hidden_size=1.0, gru_n_layers=2,
                 head_n_layers=2, enabled_tasks=None, checkpoints_dir=None, patience=2, gru_dropout=0,
                 learn_h0=False, h0_stdev=0.3, reweight_losses=False, data_version=0,
                 should_instance_val_metric_indices=True, should_instance_sku_embedder=True):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(gru_input_size, float): gru_input_size = int(self.sku_embedding_size * gru_input_size)
        if isinstance(gru_hidden_size, float): gru_hidden_size = int(gru_hidden_size * gru_input_size)

        self.sku_embedder_fname = sku_embedder_fname
        self.max_epochs = max_epochs
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gru_dropout = gru_dropout
        self.gru_input_size = gru_input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_n_layers = gru_n_layers
        self.head_n_layers = head_n_layers
        self.enabled_tasks = enabled_tasks
        self.checkpoint_dir = checkpoints_dir
        self.patience = patience
        self.item_adapter_depth = item_adapter_depth
        self.learn_h0 = learn_h0
        self.h0_stdev = h0_stdev
        self.reweight_losses = reweight_losses
        self.data_version = data_version
        self.is_training = False
        self.should_instance_val_metric_indices = should_instance_val_metric_indices
        self.should_instance_sku_embedder = should_instance_sku_embedder

        if reweight_losses:
            self.log_var = nn.Parameter(
                torch.zeros(8 if enabled_tasks is None else len(enabled_tasks)).to(device),
                requires_grad=True
            )
        else:
            self.log_var = None

        if self.should_instance_sku_embedder:
            self.sku_embedder = sku_embedder = self.make_sku_embedder(device, sku_embedder_fname)
            # cannot train if no sku embedder
            self.loss_fn = self.make_loss(enabled_tasks, sku_embedder)

        adapter_steps = []
        # Do not create adapter if not necessary
        if self.sku_embedding_size != gru_hidden_size:
            adapter_steps.append(nn.Linear(self.sku_embedding_size, gru_input_size))

        for i in range(item_adapter_depth - 1):
            adapter_steps.append(ResidualBlock(gru_input_size))

        if adapter_steps:
            self.item_adapter = nn.Sequential(*adapter_steps).to(device)
        else:
            self.item_adapter = None

        self.instance_recurrent_model()

        head_factories = self.get_head_factories()
        if self.enabled_tasks is None:
            self.enabled_tasks = list(head_factories.keys())

        heads = {}
        for t in self.enabled_tasks:
            heads[t] = head_factories[t]()
        self.heads = nn.ModuleDict(heads)

        if should_instance_val_metric_indices:
            self.instance_val_metric_indices(device)

    def on_epoch_end(self):
        self.is_training = True

    def _ensure_batch_on_device(self, batch):
        # This is needed to evaluate on training
        if self.device != 'cpu':
            seq_lengths, evts, tasks, masks = batch
            if evts.device.type != self.device:
                evts = evts.to(self.device)
                tasks = {k: v.to(self.device) for k, v in tasks.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}
                batch = (seq_lengths, evts, tasks, masks)
        return batch

    def to(self, device):
        res = super().to(device)
        if self.should_instance_val_metric_indices:
            self.instance_val_metric_indices(device)
        self.sku_embedder.to(device)
        return res

    def make_loss(self, enabled_tasks, sku_embedder):
        return MultiTaskLoss(enabled_tasks, sku_embedder.attr_embs,
                             special_tasks=('this_purch', 'will_purch'),
                             log_var=self.log_var, price_classification=self.price_classification)

    def make_sku_embedder(self, device, sku_embedder_fname):
        attr_emb = AttributeEmbedding(sku_embedder_fname, matrices=self.attributes).load_data().to(device)
        sku_embedder = FrozenEmbedder(attr_emb)
        return sku_embedder

    def instance_val_metric_indices(self, device):
        self.descr_prec_index = self.sku_embedder.get_prec_index('descr', device, exact=True)
        self.cat_prec_index = self.sku_embedder.get_prec_index('cat', device, exact=True)

    def get_head_factories(self):
        head_factories = {
            'next_sku_descr': lambda: make_head(self.gru_hidden_size, self.descr_size, self.device, self.head_n_layers),
            'next_sku_price': lambda: make_head(self.gru_hidden_size,
                                                10 if self.price_classification else self.price_size,
                                                self.device, self.head_n_layers),
            'next_sku_cat': lambda: make_head(self.gru_hidden_size, self.cat_size, self.device, self.head_n_layers),

            'purch_sku_descr': lambda: make_head(self.gru_hidden_size, self.descr_size, self.device,
                                                 self.head_n_layers),
            'purch_sku_price': lambda: make_head(self.gru_hidden_size,
                                                 10 if self.price_classification else self.price_size,
                                                 self.device, self.head_n_layers),
            'purch_sku_cat': lambda: make_head(self.gru_hidden_size, self.cat_size, self.device, self.head_n_layers),

            'will_purch': lambda: make_head(self.gru_hidden_size, 1, self.device, self.head_n_layers),
            'this_purch': lambda: make_head(self.gru_hidden_size, 1, self.device, self.head_n_layers),
        }
        return head_factories

    def instance_recurrent_model(self):
        self.rnn = GRU(
            dropout=self.gru_dropout,
            input_size=self.gru_input_size, hidden_size=self.gru_hidden_size,
            num_layers=self.gru_n_layers, batch_first=True
        ).to(self.device)
        if self.learn_h0:
            self.h0 = Parameter(
                # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
                torch.normal(torch.tensor(0.), torch.tensor(self.h0_stdev),
                             size=(self.gru_n_layers, self.gru_hidden_size)).to(self.device),
                requires_grad=True
            )

    def forward(self, batch, lengths):
        embedded_seq_tensor = self.sku_embedder(batch)
        if self.item_adapter is not None:
            embedded_seq_tensor = self.item_adapter(embedded_seq_tensor)
        packed_input = pack_padded_sequence(embedded_seq_tensor, lengths.cpu().numpy(), batch_first=True)
        packed_output = self.do_recurrent_forward(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        return output, {k: h(output) for k, h in self.heads.items()}

    def do_recurrent_forward(self, packed_input):
        if self.learn_h0:
            h0 = self.h0[:, None, :].repeat(1, packed_input.batch_sizes[0], 1)
        else:
            h0 = None
        packed_output, _ = self.rnn(packed_input, h0)
        return packed_output

    def training_step(self, batch, batch_nb):
        loss, task_losses, _ = self._step(batch)

        self.log(f"tr_loss", loss, prog_bar=True)
        for task_name, task_loss in task_losses.items():
            self.log(f"{task_name}_tr_loss", task_loss, prog_bar=False)

        if self.learn_h0:
            self.log('h0_norm', self.h0.norm(), prog_bar=False)

        if self.reweight_losses:
            for i, v in enumerate(self.log_var):
                self.log('log_var_{}'.format(self.enabled_tasks[i]), v, prog_bar=False)
        return loss

    def eval_on_training_set(self):
        n_batches = len(self.val_dataloader())
        # TODO: refactor methods
        self.on_validation_epoch_start()
        outputs = []
        for i, batch in enumerate(islice(self.train_dataloader(), n_batches if self.is_training else 2)):
            batch = self._ensure_batch_on_device(batch)
            outputs.append(self.validation_step(batch, i))
        self.validation_epoch_end(outputs, 'tr_eval', eval_on_training_set=False)

    def on_validation_epoch_start(self) -> None:
        # This is due to the fact that pytorch lightning does not handle this many losses
        self.val_task_losses = defaultdict(list)

        attr_metric_factory = self.get_attr_metric_factories()

        self.val_metric_accumulators = {}
        for (task_family, attr), factory in attr_metric_factory.items():
            task = f'{task_family}_{attr}'
            if task not in self.enabled_tasks: continue
            self.val_metric_accumulators[task_family, attr] = factory()

        self.binary_metric_accumulators = {
            task_name: RocAUCAccumulator()
            for task_name in ['will_purch', 'this_purch']
            if task_name in self.enabled_tasks
        }
        self.val_start_time = time()
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Starting validation')

    def get_attr_metric_factories(self):
        attr_metric_factory = {
            ('next_sku', 'descr'): lambda: self.descr_prec_index.prec_at_k_accum(20),
            ('purch_sku', 'descr'): lambda: self.descr_prec_index.prec_at_k_accum(20),
            ('next_sku', 'cat'): lambda: self.cat_prec_index.prec_at_k_accum(20),
            ('purch_sku', 'cat'): lambda: self.cat_prec_index.prec_at_k_accum(20),
            ('next_sku', 'price'): lambda: PriceMetricsAccumulator(),
            ('purch_sku', 'price'): lambda: PriceMetricsAccumulator(),
        }
        return attr_metric_factory

    def validation_step(self, batch, batch_nb):
        loss, task_losses, head_activations = self._step(batch)
        for tn, tl in task_losses.items():
            self.val_task_losses[tn].append(tl)

        lengths, sessions, tasks, masks = batch
        for (task_family_name, attr), accum in self.val_metric_accumulators.items():
            task_name = f'{task_family_name}_{attr}'
            pred = head_activations[task_name][masks[task_family_name]]

            if attr in self.sku_embedder.attributes:
                target = tasks[task_family_name][masks[task_family_name]]
                target = self.sku_embedder.attr_embs.emb(attr, target)
            else:
                target = tasks[task_name][masks[task_family_name]]

            accum.add_stats(pred, target)

        for task_name, accum in self.binary_metric_accumulators.items():
            pred = head_activations[task_name]
            target = tasks[task_name]
            accum.add_stats(pred, target)

        return float(loss)

    def validation_epoch_end(self, outputs, prefix='val', eval_on_training_set=True) -> None:
        self.log(f"{prefix}_loss", np.mean(outputs), prog_bar=True)
        for task_name, task_loss in self.val_task_losses.items():
            self.log(f"{task_name}_{prefix}_loss", np.mean(task_loss), prog_bar=False)

        for (task_family_name, attr), accum in self.val_metric_accumulators.items():
            accum.log_metrics(self, task_name=f'{prefix}_{task_family_name}_{attr}')

        for task_name, accum in self.binary_metric_accumulators.items():
            accum.log_metrics(self, f'{prefix}_{task_name}')

        print(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Validation time {:.02f}s'.format(time() - self.val_start_time)
        )

        if eval_on_training_set:
            with timeit('validate on training set'):
                self.eval_on_training_set()

    def _step(self, batch):
        batch = self._ensure_batch_on_device(batch)
        lengths, sessions, tasks, masks = batch

        output, head_activations = self(sessions, lengths)
        loss, task_losses = self.loss_fn(head_activations, tasks, masks)
        task_losses = {k: float(v.detach().cpu().numpy()) for k, v in task_losses.items()}

        return loss, task_losses, head_activations

    def configure_callbacks(self):
        return [
            LitProgressBar(),
            EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
            ModelCheckpoint(dirpath=self.checkpoint_dir, save_top_k=3, monitor="val_loss"),
            LearningRateMonitor(logging_interval='epoch')
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': MultiStepLR(
                    optimizer,
                    milestones=[int(self.max_epochs * 0.6), int(self.max_epochs * 0.9)],
                    gamma=0.1
                ),
                'interval': 'epoch',
                'frequency': 1
            }
        }

    @memoized_property
    def data(self):
        fname = self.get_data_fname()

        data = list(iter_jl(fname))
        data = [e for e in data if len(e) < 40]

        tr_data, val_data = train_test_split(data, random_state=42, test_size=0.05)

        if self.data_size is not None:
            print('Will subsample the data!')
            assert 0 < self.data_size < 1
            tr_data = self._subsample(tr_data)
        #             val_data = self._subsample(val_data)

        return tr_data, val_data

    def get_data_fname(self):
        path = fs.join(sigir_data_dir, 'train')
        if self.data_version == 0:
            fname = fs.join(path, 'tr_encoded_data.jl')
        else:
            fname = fs.join(path, f'tr_encoded_data_v{self.data_version}.jl')
        return fname

    def _subsample(self, data):
        view = []
        non_view = []
        for sess in data:
            l = non_view if sess[-1]['product_action'] != 'detail' else view
            l.append(sess)
        rnd = Random()

        return (
                rnd.sample(view, int(len(view) * self.data_size)) + non_view
        )

    def train_dataloader(self):
        tr_data = self.data[0]
        positive_distr = Counter([e[-1]['product_action'] != 'detail' for e in tr_data])
        label_weight = {k: 0.5 / v for k, v in positive_distr.items()}
        weights = [label_weight[e[-1]['product_action'] != 'detail'] for e in tr_data]

        return DataLoader(
            SessionDataset(tr_data),
            batch_size=self.batch_size, num_workers=8,
            collate_fn=BatchCollator().collate,
            shuffle=False, sampler=WeightedRandomSampler(weights, num_samples=len(tr_data)),
            #     shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        val_data = self.data[1]

        return DataLoader(
            SessionDataset(val_data),
            batch_size=self.batch_size, num_workers=8,
            collate_fn=BatchCollator().collate,
            #     shuffle=False, sampler=WeightedRandomSampler(weights, num_samples=len(data))
            shuffle=False,
            pin_memory=False,
        )


class RollingAverageMultiTaskLanguageModel(MultiTaskLanguageModel):
    def instance_recurrent_model(self):
        return

    def do_recurrent_forward(self, packed_input):
        input, lengths = pad_packed_sequence(packed_input, batch_first=True)
        denominators = torch.arange(1, input.shape[1] + 1)[None, :, None].repeat(input.shape[0], 1, 1)
        if self.device != 'cpu': denominators = denominators.to(self.device)
        res = input.cumsum(1) / denominators
        return pack_padded_sequence(res, lengths, batch_first=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        return self.layer_norm(self.nn(x) + x)


def make_head(in_dim, out_dim, device, n_layers):
    steps = []
    for i in range(n_layers - 1):
        steps.append(ResidualBlock(in_dim))
        # steps.append(nn.Linear(in_dim, in_dim))
        # steps.append(nn.LayerNorm(in_dim))
        # steps.append(nn.ReLU())
    # steps.append(nn.LayerNorm(in_dim))
    steps.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*steps).to(device)
