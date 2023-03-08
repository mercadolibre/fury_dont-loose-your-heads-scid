import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from scid.utils import fs
from scid.settings import sigir_data_dir
from scid.model.mt_model import RollingAverageMultiTaskLanguageModel

device = 'cuda'
batch_size = 4000

max_epochs = 200
data_size = 0.25
sku_embedder_fname = fs.join(sigir_data_dir, 'sku_embeddings_avg_desc.pkl')

gru_hp = {
    'data_version': 2,
    'gru_n_layers': 1,
    'h0_stdev': 0.03,
    'head_n_layers': 2,
    'item_adapter_depth': 0,
    'learn_h0': True,
    'learning_rate': 0.001,
    'reweight_losses': True,
    'gru_input_size': 256,
    'gru_hidden_size': 256,
}

rolling_hp = {
    'data_version': 2,
    'gru_input_size': 128,
    'head_n_layers': 1,
    'item_adapter_depth': 3,
    'learning_rate': 0.001,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['gru', 'rolling'])

    # Parse the command-line arguments
    args = parser.parse_args()

    logger = CSVLogger(fs.join(sigir_data_dir, 'logs', args.model_type))
    hp = gru_hp if args.model_type == 'gru' else rolling_hp
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
    )

    trainer.fit(model)


if __name__ == '__main__': main()