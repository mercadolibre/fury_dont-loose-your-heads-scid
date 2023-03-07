import json
import pickle as pkl
from random import Random

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import ParameterGrid

import fs
from .descriptor import UserNavDescriptor, CIDDescriptor
from .model import MultiTaskLanguageModel, RollingAverageMultiTaskLanguageModel
from .settings import sigir_data_dir


class GridSearch:
    def __init__(self, exp_name, grid: ParameterGrid, device, batch_size, data_size, max_epochs=250):
        self.exp_name = exp_name
        self.device = device
        self.batch_size = batch_size
        self.data_size = data_size
        self.max_epochs = max_epochs

        self.grid = list(grid)[:]
        Random(42).shuffle(self.grid)

        self.trials_dir = fs.ensure_exists(fs.join(sigir_data_dir, 'trials', exp_name))
        self.logs_dir = fs.ensure_exists(fs.join(sigir_data_dir, 'logs', exp_name))

        grid_fname = fs.join(self.trials_dir, 'grid_search.pkl')
        print('trials_dir', self.trials_dir)
        print('logs_dir', self.logs_dir)

        if fs.exists(grid_fname):
            with open(grid_fname, 'rb') as f:
                saved_grid = pkl.load(f)

            if grid.param_grid != saved_grid.param_grid:
                raise RuntimeError('Grid changed')
        else:
            with open(grid_fname, 'wb') as f:
                pkl.dump(grid, f)

        # if fs.ls(self.trials_dir) or fs.ls(self.logs_dir):
        #     print(self.trials_dir)
        #     print(self.logs_dir)
        #     raise RuntimeError('non empty')

    def post_process_hp(self, hp):
        return hp

    def run(self):
        for trial_id, hp in enumerate(self.grid):
            hp = self.post_process_hp(hp)
            self.run_trial(trial_id, hp)

    def run_trial(self, trial_id, hp):
        trial_fname = fs.join(self.trials_dir, f'{trial_id}.json')
        if fs.exists(trial_fname):
            with open(trial_fname) as f:
                trial = json.load(f)
                metrics_fname = fs.join(trial['metrics'], 'metrics.csv')
                if fs.exists(metrics_fname):
                    print(f'skipping trial {trial_id}...')
                    return

        model = self.instance_model(hp)
        print(hp)
        print(model)

        logger = CSVLogger(self.logs_dir)
        print('will log into ', logger.log_dir)
        with open(trial_fname, 'w') as f:
            json.dump({'hp': hp, 'metrics': logger.log_dir}, f)

        trainer = Trainer(
            gpus=1,
            max_epochs=self.max_epochs,
            logger=logger,
            log_every_n_steps=10,
            check_val_every_n_epoch=3,
            enable_progress_bar=False
        )

        trainer.fit(model)

    def instance_model(self, hp):
        raise NotImplementedError()


class GRUGridSearch(GridSearch):
    def post_process_hp(self, hp):
        hp['gru_input_size'] = hp['gru_hidden_size'] = hp.pop('gru_size')
        return hp

    def instance_model(self, hp):
        sku_embedder_fname = fs.join(sigir_data_dir, 'train/sku_embeddings_avg_desc.pkl')
        model = MultiTaskLanguageModel(
            sku_embedder_fname, self.device, max_epochs=self.max_epochs,
            data_size=self.data_size, batch_size=self.batch_size, **hp
        )
        return model


class RollingGridSearch(GridSearch):
    def instance_model(self, hp):
        sku_embedder_fname = fs.join(sigir_data_dir, 'train/sku_embeddings_avg_desc.pkl')
        model = RollingAverageMultiTaskLanguageModel(
            sku_embedder_fname, self.device, max_epochs=self.max_epochs,
            data_size=self.data_size, batch_size=self.batch_size, **hp
        )
        return model


def load_trials(path):
    trials = []
    for fname in fs.glob(fs.join(path, '*.json')):
        with open(fname) as f:
            trial = json.load(f)
        metrics_fname = fs.join(trial['metrics'], 'metrics.csv')
        if not fs.exists(metrics_fname):
            metrics_fname = fs.join(trial['metrics'], 'metrics.csv')
        if not fs.exists(metrics_fname):
            continue
        trial['df'] = pd.read_csv(metrics_fname)
        if 'val_loss' not in trial['df']: continue
        #     trial['df'].index = trial['df'].index / len(trial['df'])
        trials.append(trial)
    print(len(trials))
    return trials


SIGIR_VAL_COLS = [
    'val_next_sku_descr_prec_at_1',
    'val_purch_sku_descr_prec_at_1',
    'val_next_sku_cat_prec_at_1',
    'val_purch_sku_cat_prec_at_1',
    'val_purch_sku_price_acc',
    'val_next_sku_price_acc',
    'val_will_purch_auc', 'val_this_purch_auc'
]


def get_trial_metrics_summary(trial, mode='max'):
    d = {}
    df = trial['df']
    if 'val_loss' not in df.columns: return

    if 'val_purch_sku_price_mape' in df.columns:  # meli
        df['val_purch_sku_price_acc'] = 1 - df.val_purch_sku_price_mape.clip(upper=1)
        df['val_next_sku_price_acc'] = 1 - df.val_next_sku_price_mape.clip(upper=1)

    for c in df.columns:
        if mode == 'max':
            d[c] = df[c].dropna().max()
        elif mode == 'last':
            d[c] = df[c].dropna().iloc[-1]

    d['min_val_loss'] = df.val_loss.dropna().min()
    d['overfit'] = df.val_loss.dropna().iloc[-1] / df.tr_loss.dropna().iloc[-1]
    d.update(trial['hp'])
    d['log'] = trial['metrics']
    return d


def build_runs_df(trials, mode='max', val_cols=None):
    val_cols = val_cols or SIGIR_VAL_COLS
    assert mode in ('max', 'last')

    runs_df = []
    for trial in trials:
        d = get_trial_metrics_summary(trial, mode)
        if d is None: continue
        runs_df.append(d)

    runs_df = pd.DataFrame(runs_df)
    runs_df['score'] = 2 * runs_df[val_cols].prod(axis=1) / runs_df[val_cols].sum(axis=1)
    runs_df = runs_df.sort_values('score', ascending=False)
    return runs_df


def get_checkpoints_metrics(run, run_df):
    metrics = {}
    for checkpoint_fname in fs.ls(run['log'], 'checkpoints'):
        step = int(fs.strip_ext(fs.name(checkpoint_fname)).split('-')[1].split('=')[1])
        metrics[checkpoint_fname] = run_df[SIGIR_VAL_COLS + ['score']].loc[step].dropna().iloc[0]
    return metrics


def load_descriptors(cat2vec_fname, mtl_cls, run, pooling_strategy, device='cuda', batch_size=4096):
    run_df = get_run_df(run)
    checkpoint_metrics = get_checkpoints_metrics(run, run_df)
    checkpoint_fname, _ = max(checkpoint_metrics.items(), key=lambda x: x[1].score)

    desc = UserNavDescriptor(
        checkpoint_fname, cat2vec_fname, pooling_strategy=pooling_strategy,
        verbose=True, device=device, mtl_cls=mtl_cls, batch_size=batch_size
    )
    desc.sync_resources()
    cid_desc = CIDDescriptor(
        checkpoint_fname, cat2vec_fname, item_key='candidate',
        verbose=True, device=device, mtl_cls=mtl_cls, batch_size=batch_size
    )
    cid_desc._model = desc._model
    return cid_desc, desc, checkpoint_fname


def get_run_df(run):
    df = pd.read_csv(fs.join(run['log'], 'metrics.csv')).set_index('step')
    df['score'] = 2 * df[SIGIR_VAL_COLS].prod(axis=1) / df[SIGIR_VAL_COLS].sum(axis=1)
    return df
