import numpy as np
import hydra
import torch
import logging
from torchvision.utils import make_grid
from utils import inf_loop, MetricTracker

log = logging.getLogger(__name__)


class Trainer():
    """
    Trainer class
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initalizae_trainer()

    def _initalize_trainer(self):
        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn
        if not self.has_training:
            self._cfg.training = self._cfg

        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0

        # # Checkpoint
        # self._checkpoint: ModelCheckpoint = ModelCheckpoint(
        #     self._cfg.training.checkpoint_dir,
        #     self._cfg.model_name,
        #     self._cfg.training.weight_name,
        #     run_config=self._cfg,
        #     resume=bool(self._cfg.training.checkpoint_dir),
        # )
        #
        # # Create model and datasets
        # if not self._checkpoint.is_empty:
        #     self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
        #     self._model: BaseModel = self._checkpoint.create_model(
        #         self._dataset, weight_name=self._cfg.training.weight_name
        #     )
        # else:
        #     self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
        #     if not self._checkpoint.validate(self._dataset.used_properties):
        #         log.warning(
        #             "The model will not be able to be used from pretrained weights without the corresponding dataset."
        #         )
        #     self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
        #     self._model.instantiate_optimizers(self._cfg)

        # log.info(self._model)
        # self._model.log_optimizers()
        # log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        # self._dataset.create_dataloaders(
        #     self._model,
        #     self._cfg.training.batch_size,
        #     self._cfg.training.shuffle,
        #     self._cfg.training.num_workers,
        #     self.precompute_multi_scale,
        # )
        # log.info(self._dataset)

        # Run training / evaluation
        # self._model = self._model.to(self._device)
        # if self.has_visualization:
        #     self._visualizer = Visualizer(
        #         self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
        #     )



    # @property
    # def profiling(self):
    #     return getattr(self._cfg.debugging, "profiling", False)
    #
    # @property
    # def num_batches(self):
    #     return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", True)

    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", False)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    # @property
    # def wandb_log(self):
    #     if getattr(self._cfg, "wandb", False):
    #         return getattr(self._cfg.wandb, "log", False)
    #     else:
    #         return False

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.tensorboard, "log", False)
        else:
            return False

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)