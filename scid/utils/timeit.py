import os
from contextlib import contextmanager
from datetime import datetime
from time import time


class TimeObject:
    def __init__(self, name):
        self.name = name
        self.value = None


@contextmanager
def timeit(txt=None, silent=False):
    txt = txt or 'measure time'
    if not silent:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} pid {os.getpid()} | Starting to {txt}')

    t0 = time()
    to = TimeObject(txt)
    yield to
    to.value = time() - t0

    if not silent:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} pid {os.getpid()} | Time for {txt} is '
              f'{humanize_delta(to.value)}')


def humanize_delta(delta):
    units = ['seconds', 'minutes', 'hours', 'days', 'weeks']
    factors = [60, 60, 24, 7]
    current_unit = 0
    delta = float(delta)
    while current_unit < len(factors) and delta > factors[current_unit]:
        delta /= factors[current_unit]
        current_unit += 1

    if abs(delta - round(delta)) < 0.02:
        template = '{} {}'
        delta = round(delta)
    else:
        template = '{:.02f} {}'
    return template.format(delta, units[current_unit])
