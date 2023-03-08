import gzip
import os
import tempfile
import json
from scid.utils import fs


class iter_jl:
    def __init__(self, fname, compressed=None, limit=None):
        self.fname = fname
        self.compressed = compressed
        self.limit = limit

        self.stream = None
        self._it = None

    def __next__(self):
        if self._it is None:
            self._it = self._iterator()
        return next(self._it)

    def __iter__(self):
        if self._it is None:
            self._it = self._iterator()
        return self._it

    def _setup_iter(self):
        self.stream = fs.smart_open(self.fname, 'rb', compressed=self.compressed)

    def _iterator(self):
        if self.stream is None: self._setup_iter()
        limit = self.limit
        try:
            for i, line in enumerate(self.stream):
                if limit is not None and limit == i: break
                yield json.loads(line)
        finally:
            self.stream.close()


class JlWriter:
    def __init__(self, fname, tmp_dir=None, use_tmp_file=True):
        self.fname = fname
        self.use_tmp_file = use_tmp_file
        self.is_gz = fname.endswith('.gz')

        open_func = gzip.open if self.is_gz else open
        if use_tmp_file:
            if tmp_dir is not None and not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
            self.tmp_fname = tempfile.mktemp(dir=tmp_dir)
            self.tmp_stream = open_func(self.tmp_fname, 'w')
        else:
            self.tmp_stream = open_func(self.fname, 'w')

        self.is_empty = True
        self.finished = False

    def write_doc(self, doc):
        if not self.is_empty:
            self.tmp_stream.write(b'\n' if self.is_gz else '\n')

        sdoc = json.dumps(doc)

        if self.is_gz: sdoc = sdoc.encode('utf8')
        self.tmp_stream.write(sdoc)
        self.is_empty = False

    def flush(self):
        self.tmp_stream.flush()

    def finish(self):
        self.tmp_stream.close()
        if self.use_tmp_file:
            fs.move(self.tmp_fname, self.fname)
        self.finished = True

    def cleanup(self):
        # finish but dont write on the self.fname
        self.tmp_stream.close()
        if self.use_tmp_file:
            os.unlink(self.tmp_fname)
        self.finished = True

    def __del__(self):
        if hasattr(self, 'tmp_fname') and self.tmp_fname is not None and os.path.exists(self.tmp_fname):
            os.unlink(self.tmp_fname)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if exc_type is None:
            self.finish()
        else:
            self.cleanup()


def write_jl(contents, fname):
    with JlWriter(fname) as writer:
        for doc in contents:
            writer.write_doc(doc)
