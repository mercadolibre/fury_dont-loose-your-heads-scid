"""
EmbeddedDb is a wrapper for lmdb database tool
"""
import os
import lmdb
import ujson as json

from functools import lru_cache

MAX_READERS_ENV_NAME = "EDB_MAX_READERS"
MAX_READERS = 2048


class EmbeddedDB:
    def __init__(self, db_path, map_size=10**12, lock=True):
        max_readers = int(os.getenv(MAX_READERS_ENV_NAME, MAX_READERS))
        if lock:
            self.env = lmdb.open(db_path, map_size, max_readers=max_readers)
        else:
            self.env = lmdb.open(db_path, map_size, lock=False)

    def close(self):
        self.env.close()

    def size(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']

    def _max_readers(self):
        return self.env.max_readers()

    def _from_db_key(self, key):
        return key.decode()

    def _to_db_key(self, key):
        return key.encode()

    def _from_db_value(self, value):
        return json.loads(value.decode())

    def _to_db_value(self, value):
        return json.dumps(value).encode()

    @lru_cache(maxsize=512)
    def _get(self, key):
        """ Internal get method to be able to use cache """

        with self.env.begin() as txn:
            try:
                db_key = self._to_db_key(key)
            except AttributeError:
                return None
            value = txn.get(db_key)
            if value is None:
                return None
            return self._from_db_value(value)

    def get(self, key, default=None):
        """
        Gets an item from the db.
        If `key` is not found, `default` is returned.
        """

        value = self._get(key)
        if value is None:
            return default
        return value

    def put(self, key, value):
        """ Puts an item on the db. """

        with self.env.begin(write=True) as txn:
            txn.put(self._to_db_key(key), self._to_db_value(value))

    def put_bulk(self, data):
        """
        Puts an bulk of items on the db.
        `data` is either a dict or an iterable of key, value.

        For example, this two are equivalent:

            db.put_bulk([('key1', 'value1'), ('key2', 'value2')])
            db.put_bulk({'key1': 'value1', 'key2': 'value2'})
        """

        with self.env.begin(write=True) as txn:
            items = data.items() if isinstance(data, dict) else data
            for key, val in items:
                txn.put(self._to_db_key(key), self._to_db_value(val))

    def __iter__(self):
        for k, _ in self.items():
            yield k

    def items(self):
        """ Iterable that returns (key, value) for each element on the db.  """

        with self.env.begin() as txn:
            with txn.cursor() as curs:
                for k, v in curs:
                    yield self._from_db_key(k), self._from_db_value(v)

    def keys(self):
        """ Iterable that returns the key of each element on the db. """
        return self.__iter__()

    def values(self):
        """ Iterable that returns the value of each element on the db. """
        for _, v in self.items():
            yield v

    def __len__(self):
        return self.size()

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)
