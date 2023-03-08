from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scid.iters import recursive_get


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
