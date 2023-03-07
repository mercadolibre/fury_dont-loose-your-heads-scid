from collections import defaultdict
from itertools import count
import matplotlib.pyplot as plt

SIGIR_CONFIG = dict(
    attrs=['cat', 'price', 'descr'],
    tasks=['next_sku', 'purch_sku'],
    attr_metrics={
        'cat': ['prec_at_1', 'prec_at_3', 'prec_at_5'],
        'descr': ['prec_at_1', 'prec_at_3', 'prec_at_5'],
        'price': ['acc'],
    }
)


def plot_metrics(df, loss_logy, config):
    plt.figure(figsize=(20, 3))
    df = df.set_index('step')
    df['tr_loss'].dropna().plot(logy=loss_logy, logx=True, color='k', alpha=0.7)
    df['val_loss'].dropna().plot(style='-o', logy=loss_logy, logx=True, color='y', alpha=0.7)

    fig, axs = plt.subplots(3, 3, figsize=(20, 10))
    axs = axs.reshape(-1)
    plot_no = 0
    for k in reversed(sorted(df.columns, key=lambda x: 'next' in x)):
        if k in ('tr_loss', 'epoch', 'step', 'val_loss'): continue
        if 'loss' not in k: continue
        if 'val_' in k: continue

        df[k].dropna().plot(logx=False, ax=axs[plot_no], logy=loss_logy)
        df[k.replace('tr_', 'val_')].dropna().plot(logx=True, ax=axs[plot_no], logy=loss_logy, style='-o')

        axs[plot_no].set_title(k)
        plot_no += 1

    plt.tight_layout()

    plot_hr_metrics('Validation', config, df, 'val')

    plot_hr_metrics('Train', config, df, 'tr_eval')


def plot_hr_metrics(suptitle, config, df, prefix):
    attrs = config['attrs']
    tasks = config['tasks']
    attr_metrics = config['attr_metrics']

    fig, axs = plt.subplots(3, 3, figsize=(20, 10))

    for i, task in enumerate(tasks):
        for j, attr in enumerate(attrs):
            for metric_suffix in attr_metrics[attr]:
                metric = f'{prefix}_{task}_{attr}_{metric_suffix}'
                if metric not in df.columns: continue
                df[metric].dropna().plot(ax=axs[i, j], label=metric_suffix, style='-o')

            axs[i, j].legend(loc='best')
            axs[i, j].set_title(f'{task}_{attr}')
            axs[i, j].grid()

    for i, metric in enumerate(f'{prefix}_this_purch_auc {prefix}_will_purch_auc'.split()):
        if metric not in df.columns: continue
        df[metric].dropna().plot(ax=axs[2, i], label='auc', style='-o')
        axs[2, i].set_title(metric)
        axs[2, i].legend(loc='best')
        axs[2, i].grid()

    plt.suptitle(suptitle, fontweight='bold', fontsize=20)
    plt.tight_layout()


def plot_many_trains(dfs, config, loss_logy=True):
    plt.figure(figsize=(20, 5))
    for df in dfs:
        df = df.set_index('step')
        df['tr_loss'].dropna().plot(logy=loss_logy, logx=True, color='k', alpha=0.7)
        df['val_loss'].dropna().plot(style='-x', logy=loss_logy, logx=True, color='y', alpha=0.7)

    fig, axs = plt.subplots(3, 3, figsize=(20, 10))
    axs = axs.reshape(-1)
    c = count()
    loss_to_plot = defaultdict(lambda: next(c))
    for df in dfs:
        df = df.set_index('step')
        for k in reversed(df.columns):
            if k in ('tr_loss', 'epoch', 'step', 'val_loss'): continue
            if 'loss' not in k: continue
            if 'val_' in k: continue

            if k not in df.columns: continue
            plot_no = loss_to_plot[k]

            df[k].dropna().rolling(5).mean().plot(logx=True, ax=axs[plot_no], logy=loss_logy, color='k', alpha=0.8)
            df[k.replace('tr_', 'val_')].dropna().plot(style='-x', logx=True, ax=axs[plot_no], logy=loss_logy,
                                                       color='y',
                                                       alpha=0.8)

            axs[plot_no].set_title(k)

    plt.tight_layout()
    plot_many_hr_metrics(config, dfs, 'val')
    plt.suptitle('Validation', fontweight='bold', fontsize=20)
    plt.tight_layout()

    plot_many_hr_metrics(config, dfs, 'tr_eval')
    plt.suptitle('Train', fontweight='bold', fontsize=20)
    plt.tight_layout()


def plot_many_hr_metrics(config, dfs, metric_prefix):
    fig, axs = plt.subplots(3, 3, figsize=(20, 10))
    attrs = config['attrs']
    tasks = config['tasks']
    attr_metrics = config['attr_metrics']
    for df in dfs:
        df = df.set_index('step')
        for i, task in enumerate(tasks):
            for j, attr in enumerate(attrs):
                metric_suffix = attr_metrics[attr][0]
                metric = f'{metric_prefix}_{task}_{attr}_{metric_suffix}'
                if metric not in df.columns: continue
                df[metric].dropna().plot(ax=axs[i, j], label=metric_suffix, style='-', color='k', alpha=0.4)

                axs[i, j].set_title(metric)

        for i, metric in enumerate(f'{metric_prefix}_this_purch_auc {metric_prefix}_will_purch_auc'.split()):
            if metric not in df.columns: continue
            df[metric].dropna().plot(ax=axs[2, i], style='-', color='k', alpha=0.4)
            axs[2, i].set_title(metric)

    for row in axs:
        for ax in row:
            ax.grid()
