import json

import numpy as np
from itertools import islice

from .utils import fs
from .utils.embeddeddb import EmbeddedDB
from .settings import sigir_data_dir

path = fs.join(sigir_data_dir, 'train')

browsing_edb = EmbeddedDB(fs.join(path, 'browsing_train.edb'), lock=False)
sku_edb = EmbeddedDB(fs.join(path, 'sku_to_content.edb'), lock=False)


def prefix_dict(p, d, sep='_'):
    return {f'{p}{sep}{k}': v for k, v in d.items()}


def recursive_get(dict_, field):
    res = dict_
    fields = field.split('.')
    for field in fields:
        res = res[field]
    return res


def summarize_hash(h):
    return h[:4] + h[-4:]


def project(dict_, *fields):
    res = {}
    for field in fields:
        res[field] = recursive_get(dict_, field)

    return res


def parse_vector(v_str, asarray=False):
    if v_str:
        res = json.loads(v_str)
        if asarray:
            res = np.asarray(res)
        return res


def get_sku(sku):
    if not sku: return
    res = sku_edb.get(sku)
    if res is None: return
    res = res.copy()
    res['image_vector'] = parse_vector(res['image_vector'])
    res['description_vector'] = parse_vector(res['description_vector'])
    if res['price_bucket']: res['price_bucket'] = float(res['price_bucket'])
    return res


def iter_browsed(n=None, enrich=True, include_pageviews=False):
    it = browsing_edb.items()
    if n: it = islice(it, n)
    for sess_id, sess in it:
        if not include_pageviews:
            sess = [e for e in sess if e['product_sku_hash']]

        if not sess: continue
        if enrich:
            for i, e in enumerate(sess):
                e['prod'] = get_sku(e['product_sku_hash'])
        yield sess_id, sess


def iter_purchased(n=None, enrich=True, include_pageviews=False):
    it = browsing_edb.items()
    if n: it = islice(it, n)
    for sess_id, sess in it:
        for e in sess:
            if e['product_action'] == 'purchase': break
        else:
            continue
        if not include_pageviews:
            sess = [e for e in sess if e['product_sku_hash']]

        if enrich:
            for i, e in enumerate(sess):
                e['prod'] = get_sku(e['product_sku_hash'])
        yield sess_id, sess


def iter_added(n=None, enrich=True, include_pageviews=False):
    it = browsing_edb.items()
    if n: it = islice(it, n)
    for sess_id, sess in it:
        for e in sess:
            if e['product_action'] == 'add': break
        else:
            continue

        if not include_pageviews:
            sess = [e for e in sess if e['product_sku_hash']]

        if enrich:
            for i, e in enumerate(sess):
                e['prod'] = get_sku(e['product_sku_hash'])
        yield sess_id, sess


def iter_split(it, split_by, allowed_prod_acts, allow_pageviews=False):
    allowed_prod_acts = set(allowed_prod_acts)
    if not isinstance(split_by, str):
        split_by = set(split_by)
        split_func = lambda x: x in split_by
    else:
        split_func = lambda x: x == split_by

    for _, sess in it:
        sess = [e for e in sess if
                e['product_action'] in allowed_prod_acts or (allow_pageviews and e['event_type'] == 'pageview')]
        curr = []
        for e in sess:
            curr.append(e)
            if split_func(e['product_action']):
                yield curr
                curr = []
        if curr: yield curr


