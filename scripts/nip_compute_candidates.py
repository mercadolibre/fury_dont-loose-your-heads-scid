from itertools import chain
from sklearn.model_selection import ParameterGrid

from core.imports import *


def get_cocount(it, trigger_key, any_order=False, just_next=False, include_last_evt=False):
    cnt = defaultdict(Counter)
    used_sessions = 0
    for session in it:
        if not session: continue
        used_sessions += 1
        if just_next: assert not any_order

        last_ph_pos = None
        if not include_last_evt:
            for i in range(len(session) - 1, -1, -1):
                if session[i]['product_sku_hash']:
                    last_ph_pos = i
                    break

            # puro pageviews
            if last_ph_pos is None: continue

        for i, e in enumerate(session):
            if not e.get(trigger_key): continue
            for j in range(0 if any_order else i + 1, len(session)):
                if not session[j]['product_sku_hash']: continue
                if not include_last_evt and j == last_ph_pos: break

                cnt[e[trigger_key]][session[j]['product_sku_hash']] += 1

                if just_next: break
    return cnt


def get_cache_fname(co_counts_path, config):
    return fs.join(
        co_counts_path,
        '{split}_{trigger_key}'.format(**config) +
        ('_any_order' if config['any_order'] else '') +
        ('_just_next' if config['just_next'] else '') +
        ('_inc_last_evt' if config['include_last_evt'] else '') +
        '.pkl'
    )


def main():
    # Lazy import to avoid circular dependencies
    import next_item_prediction as nip

    n_stories = None
    co_counts_path = fs.ensure_exists(f'co_counts/{n_stories}')

    grid = ParameterGrid(dict(
#         split=['train', 'test', 'train_test'],
        split=['eval'],
        trigger_key=['product_sku_hash', 'hashed_url'],
        any_order=[True, False],
        just_next=[True, False],
        include_last_evt=[True, False]
    ))

    def valid_cfg(cfg):
        return not cfg['just_next'] or not cfg['any_order']

    grid = list(filter(valid_cfg, grid))

    for config in progress(grid, dyn_pi=False):
        config['n_stories'] = n_stories
        cache_fname = get_cache_fname(co_counts_path, config)
        if fs.exists(cache_fname):
            print('skipping', cache_fname)
            continue

        if config['split'] == 'train_test':
            data = chain(
                nip.get_split_iter('train', n_stories=n_stories // 2 if n_stories else None),
                nip.get_split_iter('test', n_stories=n_stories // 2 if n_stories else None)
            )
        else:
            data = nip.get_split_iter(config['split'], n_stories=n_stories if n_stories else None)

        co_counts = get_cocount(
            data, config['trigger_key'], any_order=config['any_order'],
            just_next=config['just_next'], include_last_evt=config['include_last_evt']
        )

        with open(cache_fname, 'wb') as f:
            pkl.dump(dict(config=config, co_counts=co_counts), f, pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
