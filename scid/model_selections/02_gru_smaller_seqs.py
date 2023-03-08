from sklearn.model_selection import ParameterGrid

from scid.model_selections.grid_search import GRUGridSearch

grid = ParameterGrid(dict(
    item_adapter_depth=[0, 1],
    gru_size=[101, 128, 256],
    gru_n_layers=[1, 2, 3],
    head_n_layers=[1, 2],
    learning_rate=[1e-3],
    h0_stdev=[0.03],
    learn_h0=[True],
    reweight_losses=[True],
    data_version=[2]
))

GRUGridSearch(
    exp_name='gru_smaller_sequences',
    grid=grid,
    device='cuda',
    batch_size=4096,
    data_size=0.25,
).run()
