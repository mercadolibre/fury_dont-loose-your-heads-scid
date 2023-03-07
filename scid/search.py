from datetime import datetime

from tqdm import tqdm

import fs
from settings import sigir_data_dir
from serialization import iter_jl


def load_data(limit=None, censored=False, do_binarize=True):
    path = fs.join(sigir_data_dir, 'train')
    fname = 'joined_data.jl.gz' if censored else 'joined_data_uncensored.jl.gz'
    it = iter_jl(fs.join(path, fname), limit=limit)
    data = list(tqdm(it))

    train_cut = datetime(2019, 3, 24)
    train_cut_ts = train_cut.timestamp() * 1000

    train_data = [e for e in filter_valid(data) if e['server_timestamp_epoch_ms'] < train_cut_ts]
    test_data = [e for e in filter_valid(data) if e['server_timestamp_epoch_ms'] >= train_cut_ts]

    print(len(train_data), len(test_data))

    if do_binarize:
        raw_X_train, raw_y_train = split_Xy(binarize(train_data))
        raw_X_test, raw_y_test = split_Xy(binarize(test_data))
        return raw_X_train, raw_y_train, raw_X_test, raw_y_test
    else:
        return train_data, test_data


def filter_valid(it):
    for doc in it:
        valid_session = all(not isinstance(e['price_bucket'], str) for e in doc['session'])
        if not valid_session: continue

        product_skus_hash = []
        product_skus = []
        for cand in doc['product_skus']:
            if cand['description_vector'] is None: continue
            if cand['price_bucket'] is None or cand['price_bucket'] == '': continue
            product_skus.append(cand)
            product_skus_hash.append(cand['product_sku_hash'])

        if not product_skus:
            # Todos los candidatos estan mal
            continue

        printed_skus = set(product_skus_hash)
        clicked_skus = [e for e in doc['clicked_skus_hash'] if e in printed_skus]
        if not clicked_skus: continue

        doc['product_skus'] = product_skus
        doc['product_skus_hash'] = product_skus_hash

        yield doc


def binarize(it):
    for row_id, doc in enumerate(it):
        printed_skus = set(doc['product_skus_hash'])
        clicked_skus = set([e for e in doc['clicked_skus_hash'] if e in printed_skus])
        assert clicked_skus

        any_true = False
        for i, cand in enumerate(doc['product_skus']):
            assert cand['description_vector'] is not None
            assert cand['price_bucket'] is not None and cand['price_bucket'] != ''
            # if cand['image_vector'] is None: continue

            label = cand['product_sku_hash'] in clicked_skus
            any_true = any_true or label
            yield dict(
                candidate=cand,
                position=i,
                query_vector=doc['query_vector'],
                session=doc['session'],
                label=label,
                row_id=row_id
            )

        assert any_true

def split_Xy(data):
    raw_X = []
    raw_y = []
    for doc in data:
        doc = doc.copy()
        raw_y.append(doc.pop('label'))
        raw_X.append(doc)
    return raw_X, raw_y
