import cv2
import numpy as np
import zipfile
import glob

from .. import Dataset


class ADE20K(Dataset):

    def __init__(self, root: str, split="train", transform=None, **kwargs):
        super().__init__(split=split, transform=transform)
        self._populate_ade20k_pairs()

    def __getitem__(self, index):
        img = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)[
            ..., ::-1
        ]  # Result should be RGB
        img = (
            img.astype(np.float32) / 255
        )  # Images are supposed to be float in range [0, 1]
        mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=img, mask=mask)

        return transformed["image"], transformed["mask"]

    def __len__(self):
        return len(self.images)

    def _populate_ade20k_pairs(self):
        """Populates the attributes:
        * images - list of Paths to images
        * masks - list of Paths to semantic segmentation masks
        """

        self.images, self.masks = [], []

        path = self.path + "ADEChallengeData2016"

        if self.split == "train":
            img_folder = path + "images/training"
            mask_folder = path + "annotations/training"
            expected_len = 20210
        elif self.split == "val":
            img_folder = path + "images/validation"
            mask_folder = path + "annotations/validation"
            expected_len = 2000
        else:
            raise ValueError(f'split must be train or val; got "{self.split}"')

        # potential_images = img_folder.glob("*.jpg")
        for potential_image in glob.glob(f'{img_folder}/*.jpg'):
            mask_path = mask_folder + (potential_image.stem + ".png")
            if mask_path.is_file():
                self.images.append(potential_image)
                self.masks.append(mask_path)
            else:
                print(f"Cannot find mask for {potential_image}")

        assert (
            len(self.images) == expected_len
        ), f"Expected {expected_len} exemplars, only found {len(self.images)}"
