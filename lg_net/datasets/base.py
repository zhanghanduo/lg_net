#!/usr/bin/env python
# coding: utf-8

import random

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import ROOT_DATASET_PATH, DATASETS

from PIL import Image
from torch.utils import data


class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """
    def __init__(
        self,
        ignore_label,
        mean_bgr,
        root: str,
        split="train",
        transform=None,
        annot_type='json',
        base_size=None,
        crop_size=321,
        scales=(1.0),
        flip=True,
    ):
        self.root = root
        self.split = split.lower()
        assert self.split in ("train", "val", "test")
        self.ignore_label = ignore_label
        self.mean_bgr = np.array(mean_bgr)
        self.annot_type = annot_type
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self.dataset = None
        self.nclasses = 0
        self._set_files()
        if transform is None:
            self.transform = A.Compose([ToTensorV2()])
        else:
            self.transform = transform

        cv2.setNumThreads(0)

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def _augmentation(self, image, label):
        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        image, label = self._augmentation(image, label)
        # Mean subtraction
        # image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image_id, image, label.astype(np.int64)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class Dataset(data.Dataset):

    def __init_subclass__(cls, **kwargs):
        """Automatic registration stuff"""
        super().__init_subclass__(**kwargs)

        # Register in DATASETS
        modules = cls.__module__.split(".")
        if len(modules) > 3 and modules[0] == "pugh_torch" and modules[1] == "datasets":
            d = DATASETS
            for module in modules[2:-1]:
                if module not in d:
                    d[module] = {}
                d = d[module]
            d[cls.__name__.lower()] = cls

    def __init__(self, split="train", *, transform=None, **kwargs):
        """
        Attempts to download data.
        Parameters
        ----------
        split : str
            One of {"train", "val", "test"}.
            Which data partition to use. Case insensitive.
        transform : obj
            Whatever format you want. Depends on dataset __getitem__ implementation.
            Defaults to just a ``ToTensor`` transform.
            This attribute is NOT used anywhere except in the dataset-specific
            __get__ implementation, or other parent classes of the dataset..
        """

        split = split.lower()
        assert split in ("train", "val", "test")
        self.split = split

        if transform is None:
            self.transform = A.Compose(
                [
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

        # self.path.mkdir(parents=True, exist_ok=True)
        # self._download_dataset_if_not_downloaded()
        # self._unpack_dataset_if_not_unpacked()

    def _download_dataset_if_not_downloaded(self):
        self.path.mkdir(parents=True, exist_ok=True)

        if self.unpacked or self.downloaded:
            return

        self.download()
        self.downloaded = True

    def _unpack_dataset_if_not_unpacked(self):
        if self.unpacked:
            return

        self.path.mkdir(parents=True, exist_ok=True)

        self.unpack()
        self.unpacked = True

    @property
    def path(self):
        """pathlib.Path to the root of the stored data"""

        try:
            return self.__path
        except AttributeError:
            try:
                dataset_type = self.__class__.__module__.split(".")[
                    -2
                ]  # e.x. "classification", "segmentation"
            except IndexError:
                dataset_type = "unknown"
            self.__path = ROOT_DATASET_PATH / dataset_type / self.__class__.__name__
            return self.__path

    @property
    def downloaded_file(self):
        return self.path / "downloaded"

    @property
    def downloaded(self):
        """We detect if the data has been fully downloaded by a "downloaded"
        file in the root of the data directory.
        """

        return self.downloaded_file.exists()

    @downloaded.setter
    def downloaded(self, val):
        """Touch/Delete sentinel downloaded data file"""

        if val:
            try:
                self.downloaded_file.touch(exist_ok=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Could not create sentinel "{self.downloaded_file}" file, which indicates the directory doesn\'t exist, and so the data most certainly has NOT been downloaded!'
                ) from e
        else:
            self.downloaded_file.unlink(missing_ok=True)

    def download(self):
        """Function to download data to ``self.path``.
        The directories up to ``self.path`` have already been created.
        Will only be called if data has not been downloaded.
        """

        raise NotImplementedError

    @property
    def unpacked_file(self):
        return self.path / "unpacked"

    @property
    def unpacked(self):
        """We detect if the data has been fully unpacked by a "unpacked"
        file in the root of the data directory.
        """

        return self.unpacked_file.exists()

    @unpacked.setter
    def unpacked(self, val):
        """Touch/Delete sentinel unpacked data file"""

        if val:
            try:
                self.unpacked_file.touch(exist_ok=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Could not create sentinel "{self.unpacked_file}" file, which indicates the directory doesn\'t exist, and so the data most certainly has NOT been unpacked!'
                ) from e
        else:
            self.unpacked_file.unlink(missing_ok=True)

    def unpack(self):
        """Post-process the downloaded payload.
        Typically this will be something like unpacking a tar file, or possibly
        re-arranging files.
        """

        raise NotImplementedError
