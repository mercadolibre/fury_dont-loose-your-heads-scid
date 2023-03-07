from sklearn.base import BaseEstimator, TransformerMixin

import fs
from .embeddeddb import EmbeddedDB
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


class VectorProjector(BaseEstimator, TransformerMixin):
    def __init__(self, fields, ensure_matrix=True):
        if isinstance(fields, str): fields = [fields]
        self.fields = fields
        self.ensure_matrix = ensure_matrix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = []
        for doc in X:
            row = []
            for field in self.fields:
                val = recursive_get(doc, field)
                if isinstance(val, list):
                    row.extend(val)
                elif isinstance(val, int) or isinstance(val, float):
                    # TODO que types
                    row.append(val)
                else:
                    raise ValueError()
            res.append(row)

        res = np.asarray(res)
        if self.ensure_matrix and len(res.shape) == 1:
            res = res[:, None]
        return res


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_cnt=50):
        if isinstance(fields, str): fields = [fields]
        self.min_cnt = min_cnt
        self.fields = fields

    def fit(self, X, y):
        values = defaultdict(lambda: defaultdict(list))
        for x_i, y_i in zip(X, y):
            for field in self.fields:
                val = recursive_get(x_i, field)
                values[field][val].append(y_i)

        self.values_ = {}
        self.default_value_ = {}
        for field, field_targets in values.items():
            default_value_targets = []
            field_stats = {}
            for val, targets in field_targets.items():
                if len(targets) < self.min_cnt:
                    default_value_targets.extend(targets)
                else:
                    field_stats[val] = [np.mean(targets), len(targets), 0]
            self.default_value_[field] = [np.mean(default_value_targets), len(default_value_targets), 1]
            self.values_[field] = field_stats

        return self

    def transform(self, X):
        res = []
        for doc in X:
            row = []
            for field in self.fields:
                val = recursive_get(doc, field)
                try:
                    row.extend(self.values_[field][val])
                except KeyError:
                    row.extend(self.default_value_[field])
            res.append(row)

        res = np.asarray(res)
        return res


class SessionAverageVector(BaseEstimator, TransformerMixin):
    def __init__(self, session_field, attribute_field):
        self.session_field = session_field
        self.attribute_field = attribute_field
        self.projector = VectorProjector(attribute_field, drop_nones=True)

    def fit(self, X, y=None):
        self.projector.fit(X, y)
        return self

    def transform(self, X):
        return np.asarray([self.projector.transform(e[self.session_field]).mean(0) for e in X])
