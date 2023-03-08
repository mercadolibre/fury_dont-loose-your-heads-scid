import numpy as np
import pickle as pkl
from itertools import groupby

from tqdm import tqdm

import scid.utils.transformers
from scid.utils.timeit import timeit
from ..scid import search
from scid.utils import fs
from ..scid.descriptor import PoolingStrategy
from scid.model_selections.grid_search import load_trials, build_runs_df
from ..scid.model import MultiTaskLanguageModel, RollingAverageMultiTaskLanguageModel
from ..scid.settings import sigir_data_dir
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_union, make_pipeline

from scid.model_selections.grid_search import load_descriptors

N_ESTIMATORS = 300
DATA_SIZE = 50_000


# def sample(X, y, pct=0.1):
#     indices = np.nonzero(np.random.RandomState(42).random(len(X)) < pct)[0]
#     return [X[i] for i in indices], [y[i] for i in indices]
#
#
# bin_X_train, bin_y_train = sample(bin_X_train, bin_y_train, pct=0.5)
# bin_X_test, bin_y_test = sample(bin_X_test, bin_y_test, pct=0.1)


def lgbm_factory(n_estimators):
    return LGBMClassifier(
        boosting_type='goss',
        n_estimators=n_estimators,
        learning_rate=0.01,
        num_leaves=15,
        silent=False, verbosity=0,
        n_jobs=8, force_col_wise=True
    )


def build_pipe(n_estimators, desc=None, cid_desc=None):
    steps = [
        scid.utils.transformers.VectorProjector('candidate.description_vector'),
        scid.utils.transformers.VectorProjector('query_vector'),
    ]
    if desc is not None: steps.append(desc)
    if cid_desc is not None: steps.append(cid_desc)

    return make_pipeline(
        make_union(*steps),
        lgbm_factory(n_estimators)
    )


def main(out_dir, n_stories, n_estimators):
    out_dir = fs.ensure_exists(fs.join(out_dir, f'{n_stories}stories_{n_estimators}estimators'))

    bin_X_train, bin_y_train, bin_X_test, bin_y_test = search.load_data(censored=False, limit=n_stories)
    bin_y_train = np.asarray(bin_y_train)
    bin_y_test = np.asarray(bin_y_test)

    if len(bin_y_test) == 0: # muy poca data, es solo para prueba
        print("WARNING!!! TEST CODE")
        train_data, test_data = search.load_data(censored=False, limit=10_000, do_binarize=False)
        bin_X_test = bin_X_train
        bin_y_test = bin_y_train
        test_data = train_data
    else:
        train_data, test_data = search.load_data(censored=False, limit=50_000, do_binarize=False)

    cat2vec_fname = fs.join(sigir_data_dir, 'train/cat_avg_desc_emb.pkl')
    gru_runs_df = build_runs_df(load_trials(fs.join(sigir_data_dir, 'trials/gru_smaller_sequences')))
    rolling_runs_df = build_runs_df(load_trials(fs.join(sigir_data_dir, 'trials/rolling_smaller_sequences')))

    gru_runs_df.iloc[0].to_dict()

    gru_cid, gru_desc, gru_checkpoint_fname = load_descriptors(
        cat2vec_fname, MultiTaskLanguageModel, gru_runs_df.iloc[0], PoolingStrategy.AVG, device='cuda'
    )

    roll_cid, roll_desc, roll_checkpoint_fname = load_descriptors(
        cat2vec_fname, RollingAverageMultiTaskLanguageModel, rolling_runs_df.iloc[0], PoolingStrategy.LAST,
        device='cuda'
    )

    name2cid = dict(roll=roll_cid, gru=gru_cid)
    name2hid = dict(roll=roll_desc, gru=gru_desc)

    grid = ParameterGrid(dict(
        model_name=['roll', 'gru'],
        use_hid=[True, False],
        use_cid=[True, False]
    ))

    evaluator = Evaluator(bin_X_train, bin_y_train, bin_X_test, bin_y_test, train_data, test_data)

    for config in tqdm(grid, desc='grid', dyn_pi=False):
        if not config['use_hid'] and not config['use_cid']:
            base_name = 'base_model.pkl'
        else:
            base_name = (
                    config['model_name'] +
                    ('_hid' if config['use_hid'] else '') +
                    ('_cid' if config['use_cid'] else '') +
                    '.pkl'
            )

        stats_fname = fs.join(fs.ensure_exists(fs.join(out_dir, 'stats')), base_name)
        pipe_fname = fs.join(fs.ensure_exists(fs.join(out_dir, 'pipes')), base_name)
        print(stats_fname)
        print(pipe_fname)
        print(config)
        if fs.exists(pipe_fname):
            print(f'skipping, pipe_fname exists')
            continue

        hid = name2hid[config['model_name']] if config['use_hid'] else None
        cid = name2cid[config['model_name']] if config['use_cid'] else None
        pipe = build_pipe(n_estimators, hid, cid)

        pipe.fit(bin_X_train, bin_y_train)

        stats, stats_input = evaluator.eval_pipe(pipe)

        with open(stats_fname, 'wb') as f:
            pkl.dump(dict(stats=stats, stats_input=stats_input), f, pkl.HIGHEST_PROTOCOL)

        with open(pipe_fname, 'wb') as f:
            pkl.dump(pipe, f, pkl.HIGHEST_PROTOCOL)


class Evaluator:
    def __init__(self, bin_X_train, bin_y_train, bin_X_test, bin_y_test, train_data, test_data):
        self.bin_X_train = bin_X_train
        self.bin_y_train = bin_y_train
        self.bin_X_test = bin_X_test
        self.bin_y_test = bin_y_test
        self.train_data = train_data
        self.test_data = test_data

        self.bin_was_seen_tr = self._get_was_seen_mask_bin(bin_X_train, bin_y_train)
        self.bin_was_seen_te = self._get_was_seen_mask_bin(bin_X_test, bin_y_test)

        self.was_seen_tr = self._get_was_seen_mask_raw(train_data)
        self.was_seen_te = self._get_was_seen_mask_raw(test_data)

    def _get_ranks(self, data, model):
        ranks = []
        raw_ranks = []
        for x_i in data:
            bin_x_i = list(search.binarize([x_i]))
            if not bin_x_i:
                # no hay clicks o los clicks no estan en los candidatos
                continue
            raw_rank = find(bin_x_i, lambda x: x['label'])
            if raw_rank == -1: continue
            raw_ranks.append(raw_rank)
            scores = model.predict_proba(bin_x_i)[:, 1]
            sorted_bin_x_i, _ = zip(*sorted(zip(bin_x_i, scores), key=lambda x: -x[1]))
            ranks.append(find(sorted_bin_x_i, lambda x: x['label']))
        return np.asarray(ranks), np.asarray(raw_ranks)

    def _get_was_seen_mask_raw(self, data):
        was_seen = []
        for doc in data:
            sess_phs = set(e['product_sku_hash'] for e in doc['session'])
            seen = any([e in sess_phs for e in doc['clicked_skus_hash']])
            was_seen.append(seen)
        return np.asarray(was_seen)

    def eval_pipe(self, pipe):
        with timeit('eval bin metrics'):
            y_train_pred = pipe.predict_proba(self.bin_X_train)[:, 1]
            roc_train = roc_auc_score(self.bin_y_train, y_train_pred)
            ap_train = average_precision_score(self.bin_y_train, y_train_pred)

            y_test_pred = pipe.predict_proba(self.bin_X_test)[:, 1]
            roc_test = roc_auc_score(self.bin_y_test, y_test_pred)
            ap_test = average_precision_score(self.bin_y_test, y_test_pred)

        was_seen_tr = self.bin_was_seen_tr
        was_seen_te = self.bin_was_seen_te
        with timeit('test ranks'):
            test_ranks, test_ranks_raw = self._get_ranks(self.test_data, pipe)

        with timeit('train ranks'):
            train_ranks, train_ranks_raw = self._get_ranks(self.train_data, pipe)

        stats_input = dict(
            bin_y_test=self.bin_y_test, bin_y_train=self.bin_y_train,
            y_train_pred=y_train_pred, y_test_pred=y_test_pred,
            test_ranks=test_ranks, test_ranks_raw=test_ranks_raw,
            train_ranks=train_ranks, train_ranks_raw=train_ranks_raw
        )

        stats = {
            'roc_train': roc_train, 'ap_train': ap_train,
            'roc_test': roc_test, 'ap_test': ap_test,
            'roc_train/seen': roc_auc_score(self.bin_y_train[was_seen_tr], y_train_pred[was_seen_tr]),
            'roc_train/not_seen': roc_auc_score(self.bin_y_train[~was_seen_tr], y_train_pred[~was_seen_tr]),

            'roc_test/seen': roc_auc_score(self.bin_y_test[was_seen_te], y_test_pred[was_seen_te]),
            'roc_test/not_seen': roc_auc_score(self.bin_y_test[~was_seen_te], y_test_pred[~was_seen_te]),

            'ap_train/seen': average_precision_score(self.bin_y_train[was_seen_tr], y_train_pred[was_seen_tr]),
            'ap_train/not_seen': average_precision_score(self.bin_y_train[~was_seen_tr], y_train_pred[~was_seen_tr]),

            'ap_test/seen': average_precision_score(self.bin_y_test[was_seen_te], y_test_pred[was_seen_te]),
            'ap_test/not_seen': average_precision_score(self.bin_y_test[~was_seen_te], y_test_pred[~was_seen_te]),

            'te_mrr': (1 / (1 + test_ranks)).mean(),
            'te_mrr/seen': (1 / (1 + test_ranks[self.was_seen_te])).mean(),
            'te_mrr/no_seen': (1 / (1 + test_ranks[~self.was_seen_te])).mean(),

            'te_raw_mrr': (1 / (1 + test_ranks_raw)).mean(),
            'te_raw_mrr/seen': (1 / (1 + test_ranks_raw[self.was_seen_te])).mean(),
            'te_raw_mrr/no_seen': (1 / (1 + test_ranks_raw[~self.was_seen_te])).mean(),

            'tr_mrr': (1 / (1 + train_ranks)).mean(),
            'tr_mrr/seen': (1 / (1 + train_ranks[self.was_seen_tr])).mean(),
            'tr_mrr/no_seen': (1 / (1 + train_ranks[~self.was_seen_tr])).mean(),

            'tr_raw_mrr': (1 / (1 + train_ranks_raw)).mean(),
            'tr_raw_mrr/seen': (1 / (1 + train_ranks_raw[self.was_seen_tr])).mean(),
            'tr_raw_mrr/no_seen': (1 / (1 + train_ranks_raw[~self.was_seen_tr])).mean(),
        }
        return stats, stats_input

    def _get_was_seen_mask_bin(self, X, y):
        was_seen = []
        for row_id, rows in groupby(zip(X, y), lambda x: x[0]['row_id']):
            rows = list(rows)
            seen = None
            for x_i, y_i in rows:
                if not y_i: continue
                cand_sku = x_i['candidate']['product_sku_hash']
                seen = cand_sku in [e['product_sku_hash'] for e in x_i['session']]
                break
            assert seen is not None
            was_seen.extend([seen] * len(rows))
        return np.asarray(was_seen)


def find(l, key):
    for i, e in enumerate(l):
        if key(e): return i
    return -1


if __name__ == '__main__':
    args = (
        EnvArgsParser(prefix='')
        .specify_param('OUT_DIR')
        .specify_param('N_STORIES', param_type=int)
        .specify_param('N_ESTIMATORS', param_type=int)
        .parse(to_lower=True)
    )
    main(args.out_dir, args.n_stories, args.n_estimators)
