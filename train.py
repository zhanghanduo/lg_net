#!/usr/bin/env python
import os
import logging
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import neptune

from lg_net.datasets import get_dataset
# from model.models.model import lg_model
from lg_net.utils.utils import set_seed, flatten_omegaconf, load_obj, save_useful_info
# from matplotlib import cm as cmx


def run(cfg: DictConfig, new_dir: str) -> None:
    """
    Run pytorch-lightning model
    Args:
        cfg: hydra config
        new_dir: the run path
    """
    # 0. Argument parsing and callback setting
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    cfg.callbacks.model_checkpoint.params.filepath = new_dir + cfg.callbacks.model_checkpoint.params.filepath
    callbacks = []
    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    # 1. Logger
    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            loggers.append(load_obj(logger.class_name)(**logger.params))

    # tb_logger = TensorBoardLogger(save_dir=cfg.general.logs_folder_name, name=cfg.general.run_dir)
    # csv_logger = CsvLogger()

    neptune.init('zhanghanduo/lgnet')
    neptune.create_experiment(
        name='first-test',
        params={"max_epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size.train}  # Optional,
    )

    # 2. Trainer
    trainer = pl.Trainer(
        logger=loggers,
        early_stop_callback=EarlyStopping(**cfg.callbacks.early_stopping.params),
        checkpoint_callback=ModelCheckpoint(**cfg.callbacks.model_checkpoint.params),
        callbacks=callbacks,
        **cfg.trainer,
    )
    # 3. Model
    model = load_obj(cfg.training.lightning_module_name)(hparams=hparams, cfg=cfg)
    # 4. Data Module
    dm = load_obj(cfg.training.data_module_name)(hparams=hparams, cfg=cfg)

    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model:
        # save as a simple torch model
        model_name = cfg.general.run_dir + '/saved_models/' + cfg.general.run_dir.split('/')[-1] + '.pth'
        print(model_name)
        torch.save(model.model.state_dict(), model_name)

    neptune.stop()


@hydra.main(config_path='configs/config.yaml')
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.general.logs_dir, exist_ok=True)
    new_dir = cfg.general.run_dir
    print(cfg.pretty())
    if cfg.general.log_code:
        save_useful_info(new_dir)
    run(cfg, new_dir)


if __name__ == '__main__':
    main()
