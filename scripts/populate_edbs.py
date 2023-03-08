from scid.utils.timeit import timeit
from tqdm.notebook import tqdm
from itertools import groupby
from scid.settings import sigir_data_dir
from scid.utils import fs
import csv
import pandas as pd

from scid.utils.embeddeddb import EmbeddedDB


def prepare(it, key_field):
    for doc in it:
        yield doc[key_field], doc


def browsing_iter():
    it = csv.DictReader(open(fs.join(sigir_data_dir, 'sorted_browsing_train.csv')))
    git = groupby(it, key=lambda x: x['session_id_hash'])
    for session_id_hash, sess in git:
        sess = list(sess)
        for doc in sess:
            doc['server_timestamp_epoch_ms'] = int(doc['server_timestamp_epoch_ms'])
        sess.sort(key=lambda x: x['server_timestamp_epoch_ms'])
        yield session_id_hash, sess


edb = EmbeddedDB(fs.join(sigir_data_dir, 'sku_to_content.edb'))
it = csv.DictReader(open(fs.join(sigir_data_dir, 'sku_to_content.csv')))
edb.put_bulk(tqdm(prepare(it, 'product_sku_hash'), desc='product_sku_hash'))
edb.close()

with timeit('sort sessions'):
    sessions = pd.read_csv(fs.join(sigir_data_dir, 'browsing_train.csv'))
    sessions = sessions.sort_values('session_id_hash')
    sessions.to_csv(fs.join(sigir_data_dir, 'sorted_browsing_train.csv'), index=False)

edb = EmbeddedDB(fs.join(sigir_data_dir, 'browsing_train.edb'))
edb.put_bulk(tqdm(browsing_iter(), desc='browsing_train'))
edb.close()

if __name__ == '__main__': main()
