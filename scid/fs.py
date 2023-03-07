import os
import re
import shutil
from glob import glob as local_glob
from pathlib import Path


def exists(fname):
    return Path(fname).exists()


def join(path, *dirs):
    return os.path.join(path, *dirs)


def mkdir(path, parents=False, exist_ok=False):
    return Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def change_ext(path, ext):
    return re.sub('(\.\w{1,3})+$', ext, path)


def clear_ext(path):
    return change_ext(path, '')


def abspath(path):
    return os.path.abspath(path)


def ensure_exists(path, clean=False):
    if clean:
        rmtree(path, not_exist_ok=True)
    mkdir(path, parents=True, exist_ok=True)
    return path


def ensure_not_exists(path):
    if exists(path):
        if is_dir(path):
            rmtree(path)
        else:
            remove(path)
    return path


def ensure_clean(path):
    return ensure_exists(path, clean=True)


def move(src, dst):
    shutil.move(src, dst)


def copy(src, dst):
    shutil.copy(src, dst)


def rmtree(path, not_exist_ok=False):
    if not_exist_ok and not exists(path): return
    return shutil.rmtree(path)


def remove(path, not_exist_ok=False):
    if not_exist_ok and not exists(path): return
    return os.unlink(path)


def glob(pattern):
    return sorted(local_glob(pattern))


def walk_files(path):
    """
    Does not return sub directories
    """
    for root, subdirs, fnames in os.walk(path):
        for fname in fnames:
            yield join(root, fname)


def touch(fname):
    with open(fname, 'w'): pass


def is_dir(fname):
    return os.path.isdir(fname)


def is_file(fname):
    return exists(fname) and not is_dir(fname)


def ls(*path):
    path = join(*path)
    return glob(join(path, '*'))


def name(fname):
    return fname.rstrip('/').split('/')[-1]


def parent(path, n=1):
    res = path
    for _ in range(n):
        res = os.path.dirname(res)
    return res
