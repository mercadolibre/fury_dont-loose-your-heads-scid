from scid.utils import fs
from scid.settings import sigir_data_dir
from scid.utils.serialization import iter_jl, write_jl, JlWriter
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json
from itertools import islice
import torch
from torch import nn
import pickle as pkl

pd.set_option("display.max_rows", 50)
pd.options.display.max_colwidth = 300
pd.options.display.column_space = 20
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30
pd.options.display.max_rows = 1000
