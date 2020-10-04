import os
import logging
import hydra
import torch
from omegaconf import DictConfig
from lg_net.utils.utils import set_seed, flatten_omegaconf, load_obj, save_useful_info


def run(cfg: DictConfig, new_dir: str) -> None:
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)
    dm = load_obj(cfg.training.data_module_name)(hparams=hparams, cfg=cfg)



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
