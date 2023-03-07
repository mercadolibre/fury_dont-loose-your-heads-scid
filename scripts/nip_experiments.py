import io

import torch
from torch import storage


def f(b):
    return torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
    
storage._load_from_bytes = f


from itertools import groupby

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.pipeline import make_pipeline

from ..scid.descriptor import PoolingStrategy

from ..scid.grid_search import load_trials, build_runs_df
from ..scid.model import RollingAverageMultiTaskLanguageModel, MultiTaskLanguageModel
import next_item_prediction as nip

from ..scid.settings import sigir_data_dir
from ..scid.timeit import timeit
from ..scid.utils import prefix_dict
from cachetools import LRUCache

from ..scid.grid_search import load_descriptors


def get_split_Xy(split_name, n_stories=2000):
    data = nip.get_split_iter(
        split_name, n_cands=200, n_stories=n_stories,
        do_enrich_sessions=True, do_split_target=True, add_cands=True,
        do_binarize=False
    )

    X = []
    y = []
    for doc in data:
        X.append(doc)
        y.append(doc['y_i'])
    return X, y


def load_all_descriptors(batch_size=10000):
    sigir_home = fs.join(ETL_PATH, job_name, 'SIGIR-ecom-data-challenge/')
    path = fs.join(sigir_home, 'train')
    cat2vec_fname = fs.join(path, 'cat_avg_desc_emb.pkl')

    gru_runs_df = build_runs_df(load_trials(fs.join(sigir_data_dir, 'trials/gru_smaller_sequences')))
    rolling_runs_df = build_runs_df(load_trials(fs.join(sigir_data_dir, 'trials/rolling_smaller_sequences')))

    gru_cid, gru_h, gru_checkpoint_fname = load_descriptors(
        cat2vec_fname, MultiTaskLanguageModel, gru_runs_df.iloc[0], PoolingStrategy.AVG, device='cuda',
        batch_size=batch_size
    )

    roll_cid, roll_h, roll_checkpoint_fname = load_descriptors(
        cat2vec_fname, RollingAverageMultiTaskLanguageModel, rolling_runs_df.iloc[0], PoolingStrategy.LAST,
        device='cuda', batch_size=batch_size
    )

    return dict(gru_h=gru_h, gru_cid=gru_cid, rolling_h=roll_h, rolling_cid=roll_cid)


class Evaluator:
    def __init__(self, data_version=2, n_bin_train=100_000, n_bin_test=10_000, n_eval=3000, n_negs=20, full_ev=True):
        negs_sufix = '' if n_negs == 20 else f'_{n_negs}negs'
        self.bin_X_train = (
            SerializedIterable(f'train_X_{n_bin_train}_v{data_version}{negs_sufix}.jl.gz').add_progress(desc='bin x train')
        )
        self.bin_y_train = list(SerializedIterable(f'train_y_{n_bin_train}_v{data_version}{negs_sufix}.jl.gz'))

        self.bin_X_test = list(SerializedIterable(f'test_X_{n_bin_test}_v{data_version}{negs_sufix}.jl.gz'))
        self.bin_y_test = list(SerializedIterable(f'test_y_{n_bin_test}_v{data_version}{negs_sufix}.jl.gz'))

        self.X_train, self.y_train = get_split_Xy('train', 2 * n_eval)
        self.X_test, self.y_test = get_split_Xy('test', 10 * n_eval)
        self.X_eval, self.y_eval = get_split_Xy('eval', None)

        self.was_seen_tr = self._get_was_seen_mask(self.X_train, self.y_train)
        self.was_seen_te = self._get_was_seen_mask(self.X_test, self.y_test)
        self.was_seen_eval = self._get_was_seen_mask(self.X_eval, self.y_eval)

        self.n_eval = n_eval
        self.full_ev = full_ev

        lgbm_group = []
        for row_id, instance in groupby(self.bin_X_train.add_progress(desc='group'), key=lambda x: x['row_id']):
            lgbm_group.append(len(list(instance)))
        self.lgbm_group = lgbm_group

        self.transformed_train_cache = LRUCache(5)
        self.features_pipe_cache = {}

    descs = None

    def _get_was_seen_mask(self, X, y):
        res = []
        for x_i, y_i in zip(X, y):
            res.append(y_i['product_sku_hash'] in set(e['product_sku_hash'] for e in x_i['x_i']))
        return np.asarray(res)

    @classmethod
    def get_descriptors(cls):
        if cls.descs is None:
            cls.descs = load_all_descriptors(batch_size=15_000)
        return cls.descs

    def get_features_pipe(self, user_descriptors, **features_kwargs):
        key = [str(features_kwargs)]
        key.extend(user_descriptors)
        key = '\n'.join(key)
        if key not in self.features_pipe_cache:
            self.features_pipe_cache[key] = nip.get_features_pipe(
                *[self.get_descriptors()[e] for e in user_descriptors], **features_kwargs
            )
        return self.features_pipe_cache[key]

    def do_experiment(self, user_descriptors, n_estimators, **features_kwargs):
        with timeit(silent=True) as t:
            features_pipe = self.get_features_pipe(user_descriptors, **features_kwargs)
            tr_X_gru = self.transform_train(features_pipe)
            gru_lgbm = nip.lgbm_model(n_estimators)
            with timeit('fit model'):
                gru_lgbm.fit(tr_X_gru, self.bin_y_train, group=self.lgbm_group)
            model = make_pipeline(features_pipe, gru_lgbm)

            with timeit('eval model'):
                evaluation = self.eval_model(model)

        evaluation['features_kwargs'] = features_kwargs
        evaluation['encoders'] = user_descriptors
        evaluation['model'] = model
        evaluation['n_estimators'] = n_estimators
        evaluation['total_time'] = t.value
        return evaluation

    def transform_train(self, features_pipe):
        key = str(features_pipe)
        try:
            res = self.transformed_train_cache[key]
            print('hit!!')
        except KeyError:
            with timeit('transform x_train'):
                features_pipe.fit(self.bin_X_train, self.bin_y_train)
                res = self.transformed_train_cache[key] = features_pipe.transform(self.bin_X_train)
        return res

    def eval_model(self, model):
        with timeit('binary metrics'):
            bin_y_hat_tr = model.predict(self.bin_X_train.set_limit(10_000))
            bin_y_hat_te = model.predict(self.bin_X_test[:10_000])

            res = dict(
                tr_roc=roc_auc_score(self.bin_y_train[:10_000], bin_y_hat_tr),
                te_roc=roc_auc_score(self.bin_y_test[:10_000], bin_y_hat_te),

                tr_ap=average_precision_score(self.bin_y_train[:10_000], bin_y_hat_tr),
                te_ap=average_precision_score(self.bin_y_test[:10_000], bin_y_hat_te),
            )

        res['was_seen_tr'] = self.was_seen_tr
        res['was_seen_te'] = self.was_seen_te
        res['was_seen_eval'] = self.was_seen_eval

        with timeit('mrr ev'):
            res.update(prefix_dict('ev', nip.mrr(model, self.X_eval, self.y_eval, n_stories=None if self.full_ev else self.n_eval)))

        with timeit('mrr train'):
            res.update(prefix_dict('tr', nip.mrr(model, self.X_train, self.y_train, n_stories=self.n_eval)))

        with timeit('mrr test'):
            res.update(prefix_dict('te', nip.mrr(model, self.X_test, self.y_test, n_stories=self.n_eval)))

        return res


def do_experiments(out_fname, n_bin_train, n_negs, full_ev, data_version):
    ev = Evaluator(n_bin_train=n_bin_train, n_negs=n_negs, full_ev=full_ev, data_version=data_version)
    feature_combinations = [
        ['gru_cid'],
        [],  # baseline
        ['gru_h', 'gru_cid'],
        ['gru_h'],
        ['rolling_cid'],
        ['rolling_h'],
        ['rolling_h', 'rolling_cid'],
    ]

    # feature_combinations = [
    #     [],  # baseline
    #     ['gru_cid'],
    #     ['rolling_cid'],
    # ]

    results = []
    if fs.exists(out_fname):
        with open(out_fname, 'rb') as f:
            results = pkl.load(f)
            
    cocounts = False
    for comb in feature_combinations:
        print(comb)
        for n_estimators in [100, 300, 1000, 3000, 10000]:
            for cocount_pos_target_encoder in [True]:#, False]:#, True]:
                features_kwargs = dict(cocounts=cocounts,
                                       cocount_pos_target_encoder=cocount_pos_target_encoder)
                hparams = dict(encoders=comb, n_estimators=n_estimators, features_kwargs=features_kwargs)
                if already_computed(hparams, results):
                    print('skipping...')
                    continue
                
                evaluation = ev.do_experiment(comb, n_estimators, **features_kwargs)
                print(evaluation)
                results.append(evaluation)
                with open('tmp.pkl', 'wb') as f:
                    pkl.dump(results, f)
                fs.move('tmp.pkl', out_fname)
    return results

def already_computed(hparams, results):
    found = False
    for r in results:
        for k, v in hparams.items():
            if r[k] != v: break
        else:
            found = True
            break
            
    return found
        

def main():
    full_ev = True
    data_version = 9
    for n_bin_train in [100_000]:
        for n_negs in [5]:#, 10, 20]:
            fname = f'nip_{n_bin_train}_{n_negs}negs_v{data_version}_{"full_" if full_ev else "some_"}ev.pkl'
            print(fname)
            do_experiments(
                fname, n_bin_train, n_negs,
                full_ev=full_ev, data_version=data_version
            )


if __name__ == '__main__': main()