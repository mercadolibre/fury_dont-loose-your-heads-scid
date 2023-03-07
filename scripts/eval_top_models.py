from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.env_args_parser import EnvArgsParser, Enum
from core.imports import *
from core.sigir.descriptor import UserNavDescriptor, PoolingStrategy, CIDDescriptor
from core.sigir.grid_search import load_trials, build_runs_df
from core.sigir.model import MultiTaskLanguageModel, RollingAverageMultiTaskLanguageModel

from lightgbm import LGBMClassifier
from core.sigir import utils, search
from sklearn.pipeline import make_union, make_pipeline

from sklearn.metrics import roc_auc_score, average_precision_score

from datetime import datetime

TEST = False


metric_cols = [
    'val_next_sku_descr_prec_at_1',
    'val_purch_sku_descr_prec_at_1',
    'val_next_sku_cat_prec_at_1',
    'val_purch_sku_cat_prec_at_1',
    'val_purch_sku_price_acc',
    'val_next_sku_price_acc',
    'val_will_purch_auc', 'val_this_purch_auc'
]

def main():
    args = (
        EnvArgsParser(prefix='')
            .specify_param('TRIALS_PATH')
            .specify_param('MTL_CLS', param_type=Enum('rolling', 'gru'))
            .parse(to_lower=True)
    )

    trials = load_trials(args.trials_path)
    if not trials:
        raise RuntimeError('No trials!')
    runs_df = build_runs_df(trials)

    path = fs.join(ETL_PATH, job_name, 'SIGIR-ecom-data-challenge/train')
    it = iter_jl(fs.join(path, 'joined_data.jl.gz'), limit=1000 if TEST else None)
    data = list(progress(it, desc='load data'))

    train_cut = datetime(2019, 3, 24)

    train_cut_ts = train_cut.timestamp() * 1000
    train_data = [e for e in data if e['server_timestamp_epoch_ms'] < train_cut_ts]
    test_data = [e for e in data if e['server_timestamp_epoch_ms'] >= train_cut_ts]
    
    if TEST:
        test_data = train_data
    
    print('train data', len(train_data))
    print('test data', len(test_data))

    raw_X_train, raw_y_train = search.split_Xy(search.binarize(train_data))
    raw_X_test, raw_y_test = search.split_Xy(search.binarize(test_data))

    mtl_cls = RollingAverageMultiTaskLanguageModel if args.mtl_cls == 'rolling' else MultiTaskLanguageModel
    pooling_strategy = PoolingStrategy.LAST if args.mtl_cls == 'rolling' else PoolingStrategy.AVG

    eval_runs(path, runs_df, raw_X_test, raw_X_train, raw_y_test, raw_y_train,
              mtl_cls=mtl_cls, pooling_strategy=pooling_strategy)


def eval_runs(path, runs_df, raw_X_test, raw_X_train, raw_y_test, raw_y_train, mtl_cls, pooling_strategy):
    runs = runs_df.to_dict(orient='records')
    cat2vec_fname = fs.join(path, 'cat_avg_desc_emb.pkl')
    for run in runs:
        eval_dir = fs.ensure_exists(fs.join(run['log'], 'search'))
        if fs.exists(fs.join(eval_dir, 'done')):
            print(f'skipping {eval_dir}, already processed...')
            continue

        eval_pretrained_model(eval_dir, run, raw_X_train, raw_y_train, raw_X_test, raw_y_test, cat2vec_fname, mtl_cls,
                              pooling_strategy)


def eval_pretrained_model(out_dir, run, raw_X_train, raw_y_train, raw_X_test, raw_y_test, cat2vec_fname, mtl_cls,
                          pooling_stragegy):
    print('Will eval', run['log'])
    cid_desc, desc, _ = load_descriptors(cat2vec_fname, mtl_cls, run, pooling_stragegy)

    pipe = make_pipeline(
        make_union(
            utils.VectorProjector('candidate.description_vector'),
            utils.VectorProjector('query_vector'),
            desc,
            cid_desc
        ),
        lgbm_factory()
    )
    eval_model(fs.ensure_exists(fs.join(out_dir, 'h_cid_lgbm')), pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train)

    pipe = make_pipeline(
        make_union(
            utils.VectorProjector('candidate.description_vector'),
            utils.VectorProjector('query_vector'),
            desc,
            cid_desc
        ),
        StandardScaler(),
        LogisticRegression(max_iter=500)
    )
    eval_model(fs.ensure_exists(fs.join(out_dir, 'h_cid_linear')), pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train)

    pipe = make_pipeline(
        make_union(
            utils.VectorProjector('candidate.description_vector'),
            utils.VectorProjector('query_vector'),
            cid_desc
        ),
        lgbm_factory()
    )
    eval_model(fs.ensure_exists(fs.join(out_dir, 'cid_lgbm')), pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train)

    pipe = make_pipeline(
        make_union(
            utils.VectorProjector('candidate.description_vector'),
            utils.VectorProjector('query_vector'),
            cid_desc
        ),
        StandardScaler(),
        LogisticRegression(max_iter=500)
    )
    eval_model(fs.ensure_exists(fs.join(out_dir, 'cid_linear')), pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train)

    pipe = make_pipeline(
        make_union(
            utils.VectorProjector('candidate.description_vector'),
            utils.VectorProjector('query_vector'),
            desc
        ),
        lgbm_factory()
    )
    eval_model(fs.ensure_exists(fs.join(out_dir, 'h_lgbm')), pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train)

    pipe = make_pipeline(
        make_union(
            utils.VectorProjector('candidate.description_vector'),
            utils.VectorProjector('query_vector'),
            desc
        ),
        StandardScaler(),
        LogisticRegression(max_iter=500)
    )
    eval_model(fs.ensure_exists(fs.join(out_dir, 'h_linear')), pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train)
    fs.touch(fs.join(out_dir, 'done'))


def eval_model(out_dir, pipe, raw_X_test, raw_X_train, raw_y_test, raw_y_train):
    pipe.fit(raw_X_train, raw_y_train)
    y_train_pred = pipe.predict_proba(raw_X_train)[:, 1]
    roc_train = roc_auc_score(raw_y_train, y_train_pred)
    ap_train = average_precision_score(raw_y_train, y_train_pred)

    y_test_pred = pipe.predict_proba(raw_X_test)[:, 1]
    roc_test = roc_auc_score(raw_y_test, y_test_pred)
    ap_test = average_precision_score(raw_y_test, y_test_pred)

    with open(fs.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(dict(roc_train=roc_train, roc_test=roc_test, ap_train=ap_train, ap_test=ap_test), f)

    with open(fs.join(out_dir, 'model.pkl'), 'wb') as f:
        pkl.dump(pipe, f, 2)

    with open(fs.join(out_dir, 'test_predictions.pkl'), 'wb') as f:
        pkl.dump(y_test_pred, f, 2)

    with open(fs.join(out_dir, 'train_predictions.pkl'), 'wb') as f:
        pkl.dump(y_train_pred, f, 2)


def lgbm_factory():
    return LGBMClassifier(
        boosting_type='goss',
        n_estimators=1400,
        learning_rate=0.01,
        num_leaves=15,
        silent=False, verbosity=0,
        n_jobs=8, force_col_wise=True
    )


def get_checkpoints_metrics(run, run_df):
    metrics = {}
    for checkpoint_fname in fs.ls(run['log'], 'checkpoints'):
        step = int(fs.strip_ext(fs.name(checkpoint_fname)).split('-')[1].split('=')[1])
        metrics[checkpoint_fname] = run_df[metric_cols + ['score']].loc[step].dropna().iloc[0]
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
    df['score'] = 2 * df[metric_cols].prod(axis=1) / df[metric_cols].sum(axis=1)
    return df


if __name__ == '__main__': main()