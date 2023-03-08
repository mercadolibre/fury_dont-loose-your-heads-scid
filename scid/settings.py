import os
from scid.utils import fs

ref_path = os.path.abspath(fs.join(os.path.dirname(__file__), '..'))
sigir_data_dir = os.path.join(ref_path, 'SIGIR-ecom-data-challenge/train')