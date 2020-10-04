import os
from pathlib import Path

ROOT_DATASET_PATH = Path(
    os.environ.get("DATASET_PATH", "~/.pugh_torch/datasets")
).expanduser()

# Populated automatically via pugh_torch.datasets.Dataset.__init_subclass__
DATASETS = {}

from .base import _BaseDataset, Dataset
from .torchvision import TorchVisionDataset

# import lg_net.datasets.classification
import lg_net.datasets.segmentation


def get(genre, name):
    """Gets dataset constructor from string identifiers
    Parameters
    ----------
    genre : str
        Type of dataset. e.x. "classification".
        Case insensitive
    name : str
        Name of dataset. e.x. "imagenet".
        Case insensitive
    """

    genre = genre.lower()
    name = name.lower()

    return DATASETS[genre][name]


# Alias for ``get`` function
get_dataset = get
