from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from core.imports import *
from core.sigir.model import RollingAverageMultiTaskLanguageModel

path = fs.join(ETL_PATH, job_name, 'SIGIR-ecom-data-challenge/train')
device = 'cuda'
batch_size = 4000

max_epochs = 200
data_size = 0.25
sku_embedder_fname = fs.join(path, 'sku_embeddings_avg_desc.pkl')

hp = {
    'gru_hidden_size': 128,
    'gru_input_size': 128,
    'gru_n_layers': 1,
    'head_n_layers': 4,
    'item_adapter_depth': 4,
    'learning_rate': 0.001
}

hp = {
    'item_adapter_depth':3,
    'head_n_layers':4,
    'gru_input_size':192,
    'learning_rate': 0.001
}

logger = CSVLogger('logs/rolling_avg')
print(logger.log_dir)

model = RollingAverageMultiTaskLanguageModel(
    sku_embedder_fname, device, checkpoints_dir=fs.join(logger.log_dir, 'checkpoints'),
    patience=2,
    batch_size=batch_size, data_size=data_size, max_epochs=max_epochs, **hp
)

trainer = Trainer(
    # accelerator='gpu', devices=1,
    gpus=1,
    max_epochs=max_epochs,
    logger=logger,
    enable_checkpointing=True,
    log_every_n_steps=10,
    check_val_every_n_epoch=5,
    enable_progress_bar=False,
    #     track_grad_norm=2,
    #     auto_scale_batch_size='binsearch',
    #     auto_lr_find=True
)

trainer.fit(model)