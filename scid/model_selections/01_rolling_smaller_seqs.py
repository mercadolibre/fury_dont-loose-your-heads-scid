from sklearn.model_selection import ParameterGrid

from core.sigir.grid_search import RollingGridSearch

grid = ParameterGrid(dict(
    item_adapter_depth=[0, 1, 2, 3],
    head_n_layers=[1, 2, 3, 4],
    gru_input_size=[101, 128, 192],
    learning_rate=[1e-3],
    data_version=[2]
))

RollingGridSearch(
    job_name='data/new_data',
    exp_name='rolling_smaller_sequences',
    grid=grid,
    device='cuda',
    batch_size=4096,
    data_size=0.25,
).run()
