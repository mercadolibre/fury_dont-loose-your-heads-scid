import numpy as np
import json
import pickle as pkl
from collections import Counter
from itertools import islice
import csv
from datetime import datetime
from itertools import groupby
from random import Random
from time import time

from sklearn.pipeline import make_union, make_pipeline
from tqdm import tqdm

import scid.utils.transformers
from nip_compute_candidates import get_cache_fname
from scid.utils import fs
from scid.settings import sigir_data_dir
from ..scid import utils

train_cut = datetime(2019, 3, 24)
train_cut_ts = train_cut.timestamp() * 1000
rnd = Random(42)


def load_data(fname, n):
    with open(fname) as f:
        reader = csv.DictReader(f)
        it = groupby(reader, key=lambda x: x['session_id_hash'])
        if n is not None: it = islice(it, n)
        for _, session in it:
            yield list(session)


def train_it(it):
    for e in it:
        if int(e[-1]['server_timestamp_epoch_ms']) < train_cut_ts: yield e


def test_it(it):
    for e in it:
        if int(e[0]['server_timestamp_epoch_ms']) >= train_cut_ts: yield e


def enrich_sessions(it):
    for session in it:
        for evt in session:
            if not evt['product_sku_hash']: continue
            evt.update(utils.get_sku(evt['product_sku_hash']))
        yield session


def remove_only_pageviews_sessions(it):
    removed = 0
    n = 0
    for i, sess in enumerate(it):
        n += 1
        filtered_sess = [e for e in sess if e['product_sku_hash']]
        if not filtered_sess:
            removed += 1
            continue
        yield sess

    print(f'removed {removed} ({removed / n * 100:.01f}%) sessions without any product views')


def remove_pageviews(it):
    removed = 0
    n = 0
    for sess in it:
        n += 1
        sess = [e for e in sess if e['product_sku_hash']]
        if not sess:
            removed += 1
            continue
        yield sess
    print(f'removed {removed} ({removed / n * 100:.01f}%) sessions without product views')


def remove_duplicates(it):
    for sess in it:
        curr = None
        new_sess = []
        for evt in sess:
            evt_id = evt['product_sku_hash'] if evt['product_sku_hash'] else evt['hashed_url']
            if evt_id == curr: continue
            new_sess.append(evt)
            curr = evt_id

        yield new_sess


def remove_no_description_vector(it):
    skipped = 0
    n = 0
    for sess in it:
        n += 1
        new_sess = []

        for e in sess:
            if e['product_sku_hash'] and e.get('description_vector') and e.get('price_bucket'):
                new_sess.append(e)
            elif e['event_type'] == 'pageview':
                new_sess.append(e)

        if new_sess:
            yield new_sess
        else:
            skipped += 1

    print(f'removed {skipped} ({skipped / n * 100:.01f}%) sessions without useful data')


def split_target(it):
    n_skipped = 0
    n = 0
    for sess in it:
        n += 1
        for i in range(len(sess) - 1, -1, -1):
            if sess[i]['product_sku_hash']: break
        else:
            n_skipped += 1
            continue

        if i == 0:
            n_skipped += 1
            continue

        yield dict(x_i=sess[:i], y_i=sess[i])

    print(f'couldnt create X,y for {n_skipped} ({n_skipped / n * 100:.02f}%) sessions. '
          'No product_sku_hash in the session')


def filter_seen(it):
    n_skipped = 0
    n = 0
    for doc in it:
        n += 1
        if doc['y_i']['product_sku_hash'] in set(e['product_sku_hash'] for e in doc['x_i']):
            n_skipped += 1
            continue

        yield doc

    print(f'Removed examples where y_i was seen {n_skipped} ({n_skipped / n * 100:.02f}%)')


class Candidates:
    def __init__(self, co_counts_path, configs):
        self.co_counts_path = co_counts_path
        self.configs = configs
        self.all_cocounts = None

    def sync_resources(self):
        if self.all_cocounts: return
        all_cocounts = []
        for cfg in self.configs:
            fname = get_cache_fname(self.co_counts_path, cfg)
            with open(fname, 'rb') as f:
                all_cocounts.append(pkl.load(f))
        self.all_cocounts = all_cocounts

    def get_cocount_candidates(self, x_i, n):
        self.sync_resources()
        for evt in reversed(x_i):
            for source_id, co_counts in enumerate(self.all_cocounts):
                trigger_key = co_counts['config']['trigger_key']
                co_counts = co_counts['co_counts']
                if not evt[trigger_key]: continue
                url_cands = co_counts.get(evt[trigger_key])
                if url_cands:
                    return Counter(dict(url_cands.most_common(n))), source_id

        return Counter(), None


candidates = Candidates(
    co_counts_path='co_counts/None',

    configs=[{'any_order': False,
  'include_last_evt': False,
  'just_next': True,
  'split': 'train_test',
  'trigger_key': 'product_sku_hash',
  'n_stories': None,
  'ev_mrr': 0.2402373056815533,
  'ev_hr': 0.586193605972087},
 {'any_order': True,
  'include_last_evt': False,
  'just_next': False,
  'split': 'train_test',
  'trigger_key': 'hashed_url',
  'n_stories': None,
  'ev_mrr': 0.17509220709558637,
  'ev_hr': 0.8960361895488478}]
)
#version = 8
#     configs=[{'any_order': False,
#   'include_last_evt': False,
#   'just_next': True,
#   'split': 'train_test',
#   'trigger_key': 'product_sku_hash',
#   'n_stories': None,
#   'ev_mrr': 0.2402373056815533,
#   'ev_hr': 0.586193605972087},
#  {'any_order': False,
#   'include_last_evt': True,
#   'just_next': True,
#   'split': 'test',
#   'trigger_key': 'hashed_url',
#   'n_stories': None,
#   'ev_mrr': 0.19270227179057897,
#   'ev_hr': 0.8195999675430056}]
# )
# version = 6
#     configs=[
#         {'any_order': False,
#          'include_last_evt': False,
#           'just_next': True,
#           'split': 'eval',
#           'trigger_key': 'hashed_url',
#           'n_stories': None,
#           'ev_mrr': 0.24346775310081542,
#           'ev_hr': 0.6872362869198312},
#          {'any_order': False,
#           'include_last_evt': False,
#           'just_next': True,
#           'split': 'train_test',
#           'trigger_key': 'product_sku_hash',
#           'n_stories': None,
#           'ev_mrr': 0.2402373056815533,
#           'ev_hr': 0.586193605972087},
#           {'any_order': True,
#          'include_last_evt': True,
#          'just_next': False,
#          'split': 'train',
#          'trigger_key': 'hashed_url',
#          'n_stories': None,
#          'ev_mrr': 0.16136356772650617,
#          'ev_hr': 0.8277345017851347}
#     ]
# )


def add_candidates(it, n=None):
    for doc in it:
        x_i = doc['x_i']
        doc['candidates'], doc['candidates_source'] = candidates.get_cocount_candidates(x_i, n=n)
        yield doc


class CoCountsCandidateFeatures:
    def __init__(self, candidates, flatten_with_same_score=True):
        self.candidates = candidates
        if flatten_with_same_score:
            self.candidates_pos = {}
            pos = 0
            last_score = None
            for k, score in sorted(candidates.items(), key=lambda x: -x[1]):
                if last_score is not None and score < last_score:
                    pos += 1
                self.candidates_pos[k] = pos
                last_score = score
        else:
            self.candidates_pos = {
                k: pos for pos, (k, score) in enumerate(sorted(candidates.items(), key=lambda x: -x[1]))
            }

    def get_cand_dict(self, sku):
        res = {}
        res.update(sku)
        res['co_count'] = self.candidates[sku['product_sku_hash']]
        res['normed_co_count'] = res['co_count'] / sum(self.candidates.values())
        res['total_co_views'] = sum(self.candidates.values())
        res['co_count_pos'] = self.candidates_pos[sku['product_sku_hash']]
        return res


def binarize(it, should_remove_only_pageviews=True, n_negs=5):
    rnd = Random(42)
    skipped = Counter()
    total = 0
    for row_id, doc in enumerate(it):
        total += 1
        x_i = doc['x_i']
        y_i = doc['y_i']
        x_i_candidates = doc['candidates']
        if should_remove_only_pageviews:
            for evt in x_i:
                if evt['product_sku_hash'] != '' and evt['product_sku_hash'] is not None: break
            else:
                skipped['x_i only with pageviews'] += 1
                continue

        if not y_i['description_vector'] or not y_i['price_bucket']:
            skipped['target no meta'] += 1
            continue

        # solo para las empiricas
        target_in_cands = y_i['product_sku_hash'] in x_i_candidates
        if not target_in_cands: continue

        if len(x_i_candidates) - target_in_cands == 0:  # solo tiene el positivo
            skipped['no candidates'] += 1
            continue
        negatives = list(x_i_candidates)
        rnd.shuffle(negatives)

        yielded = 0
        cand_features = CoCountsCandidateFeatures(doc['candidates'])
        # negative
        for n in negatives:
            if n == y_i['product_sku_hash']: continue
            n_sku = cand_features.get_cand_dict(utils.get_sku(n))
            if not n_sku['description_vector'] or not n_sku['price_bucket']: continue

            yield dict(
                x_i=dict(
                    session=x_i,
                    row_id=row_id,
                    candidate=n_sku,
                    source=doc['candidates_source']
                ),
                y_i=False
            )
            yielded += 1
            if yielded == n_negs: break

        # positive
        yield dict(
            x_i=dict(
                session=x_i,
                row_id=row_id,
                candidate=cand_features.get_cand_dict(utils.get_sku(y_i['product_sku_hash'])),
                source=doc['candidates_source']
            ),
            y_i=True
        )

    for reason, cnt in skipped.items():
        print(f'skipped {reason}: {cnt} ({cnt / total * 100:.02f}%)')


def get_split_iter(split_name='train', n_stories=10_000,
                   do_enrich_sessions=False,
                   do_split_target=False,
                   add_cands=False, n_cands=200,
                   do_binarize=False, n_negs=5,
                   progress_fn=None, do_filter_seen=False):
    if split_name == 'eval':
        fname = fs.join(sigir_data_dir, 'rec_test_phase_1.json')

        with open(fname) as f:
            data = [e['query'] for e in json.load(f)]

        if n_stories:
            data = data[:n_stories]
    else:
        path = fs.join(sigir_data_dir, '/train')
        fname = fs.join(path, './sorted_browsing_train.csv')
        data = load_data(fname, n_stories)
        if split_name == 'train':
            data = train_it(data)
        else:
            data = test_it(data)

    if progress_fn:
        data = progress_fn(data)

    it = remove_duplicates(data)

    if do_enrich_sessions:
        it = remove_no_description_vector(enrich_sessions(it))

    if do_split_target:
        it = split_target(it)

    if do_filter_seen:
        it = filter_seen(it)

    if add_cands:
        it = add_candidates(it, n=n_cands)

    if do_binarize:
        assert add_cands and do_split_target
        it = binarize(it, n_negs=n_negs)
    return it


class InMemoryMetadata:
    def __init__(self):
        self.meta = None

    def sync_resources(self):
        if self.meta is not None: return
        all_candidates = {}
        path = fs.join(sigir_data_dir, 'train')
        with open(fs.join(path, 'sku_to_content.csv')) as f:
            for v in csv.DictReader(f):
                if not v['description_vector']: continue
                if not v['price_bucket']: continue
                v['description_vector'] = json.loads(v['description_vector'])
                v['price_bucket'] = float(v['price_bucket'])
                id = v['product_sku_hash']
                all_candidates[id] = v
        self.meta = all_candidates

    def __getitem__(self, sku_hash):
        self.sync_resources()
        return self.meta[sku_hash]

    def __contains__(self, sku_hash):
        self.sync_resources()
        return sku_hash in self.meta


candidates_meta = InMemoryMetadata()


def model_can_be_used(session):
    for evt in session:
        if evt.get('description_vector') and evt.get('price_bucket'): return True
    return False


def recommend(model, doc, n):
    """
    model es un sklearn pipeline
    doc sale de get_split_iter. Asume que no esta binarizado
    """
    session = doc['x_i']
    session_candidates_dict = doc['candidates']
    use_model = model_can_be_used(session)

    if not use_model:
        hashes = [
            h for h, _ in sorted(session_candidates_dict.items(), key=lambda x: -x[1])
            if h in candidates_meta
        ][:n]
        return hashes, None

    session = [e for e in session if e['product_sku_hash'] and e['description_vector'] and e['price_bucket']]
    assert len(session) > 0

    cand_features = CoCountsCandidateFeatures(doc['candidates'])

    session_candidates = []
    for product_sku_hash, cnt in sorted(session_candidates_dict.items(), key=lambda x: -x[1]):
        if product_sku_hash not in candidates_meta: continue
        candidate = cand_features.get_cand_dict(candidates_meta[product_sku_hash])
        session_candidates.append(dict(session=session, candidate=candidate, product_sku_hash=product_sku_hash))

    if len(session_candidates) == 0:
        return [], []

    # LTR
    probs = model.predict(session_candidates)
    for c, p in zip(session_candidates, probs):
        c['prob'] = p
    session_candidates.sort(key=lambda x: -x['prob'])
    session_candidates = session_candidates[:n]

    return [c['product_sku_hash'] for c in session_candidates], [c['prob'] for c in session_candidates]


def mrr(model, X, y, n=20, n_stories=None):
    last_print = time()
    inv_ranks = []
    model_was_used = []
    for doc, target in tqdm(zip(X, y), tot=len(y), desc='mrr'):
        if len(inv_ranks) == n_stories: break
        if time() - last_print > 300:
            last_print = time()
            tmp_inv_ranks = np.asarray(inv_ranks)
            tmp_model_was_used = np.asarray(model_was_used)
            print(dict(
                mrr=np.mean(inv_ranks),
                mrr_model=np.mean(tmp_inv_ranks[tmp_model_was_used]),
                mrr_fallback=np.mean(tmp_inv_ranks[~tmp_model_was_used])
            ))
            
        model_was_used.append(model_can_be_used(doc['x_i']))
        hashes, probs = recommend(model, doc, n)

        try:
            inv_ranks.append(1 / (1 + hashes.index(target['product_sku_hash'])))
        except ValueError:
            inv_ranks.append(0)

    inv_ranks = np.asarray(inv_ranks)
    model_was_used = np.asarray(model_was_used)
    
    res = dict(
        mrr=np.mean(inv_ranks),
        mrr_model=np.mean(inv_ranks[model_was_used]),
        mrr_fallback=np.mean(inv_ranks[~model_was_used])
    )

    print(res)

    res.update(dict(
        inv_ranks=inv_ranks,
        model_was_used=model_was_used,
    ))
    return res


from lightgbm import LGBMRanker
def lgbm_model(n_estimators):
    return LGBMRanker(
        boosting_type='goss',
        n_estimators=n_estimators,
        learning_rate=0.01 if n_estimators < 1000 else 0.005,
        num_leaves=15,
        silent=False, verbosity=0,
        n_jobs=8, force_col_wise=True
    )


def get_features_pipe(*user_descriptors, cocounts=False, cocount_pos_target_encoder=False):
    fields = ['candidate.description_vector', 'candidate.price_bucket']
    if cocounts:
        fields.extend(
            ['candidate.co_count_pos', 'candidate.co_count',
             'candidate.normed_co_count', 'candidate.total_co_views', 'source'
        ])

    steps = [scid.utils.transformers.VectorProjector(fields)]
    steps.extend(user_descriptors)
    if cocount_pos_target_encoder:
        steps.append(scid.utils.transformers.TargetEncoder(['candidate.co_count_pos']))

    return make_union(*steps)


def create_lgbm_model(n_estimators, *user_descriptors, **features_kwargs):
    pipe = make_pipeline(
        get_features_pipe(*user_descriptors, **features_kwargs),
        lgbm_model(n_estimators)
    )

    return pipe


def main():
    args = (
        EnvArgsParser(prefix='')
        .specify_param('N_TRAIN', int, default=None)
        .specify_param('N_TEST', int, default=10_000)
        .specify_param('N_CANDS', int, default=200)
        .specify_param('N_NEGS', int, default=20)
        .parse(to_lower=True)
    )
    version = 1
    version = 2  # con co-counts pos
    version = 3  # eval hashed_url co-counts
    version = 4  # better fallbacks
    version = 5  # do not include past events unless strictly necesary
    version = 6  # flatten_with_same_score
    version = 7  # filter seen

    train_data = get_split_iter(
        'train', n_cands=args.n_cands, n_stories=args.n_train,
        do_enrich_sessions=True, do_split_target=True, add_cands=True, do_binarize=True, n_negs=args.n_negs,
    )
    with JlWriter(f'train_X_{args.n_train}_v{version}_{args.n_negs}negs.jl.gz') as X_writer:
        with JlWriter(f'train_y_{args.n_train}_v{version}_{args.n_negs}negs.jl.gz') as y_writer:
            for doc in progress(train_data, desc='write train'):
                X_writer.write_doc(doc['x_i'])
                y_writer.write_doc(doc['y_i'])

    test_data = get_split_iter(
        'test', n_cands=args.n_cands, n_stories=args.n_test,
        do_enrich_sessions=True, do_split_target=True, add_cands=True, do_binarize=True, n_negs=20
    )
    X_fname = f'test_X_{args.n_test}_v{version}_{args.n_negs}negs.jl.gz'
    y_fname = f'test_y_{args.n_test}_v{version}_{args.n_negs}negs.jl.gz'
    if fs.exists(X_fname) and fs.exists(y_fname):
        print('test dataset already cached')
        return

    with JlWriter(X_fname) as X_writer:
        with JlWriter(y_fname) as y_writer:
            for doc in progress(test_data, desc='write test'):
                X_writer.write_doc(doc['x_i'])
                y_writer.write_doc(doc['y_i'])


if __name__ == '__main__':
    main()
