import os
import json
import logging
import hydra
import numpy as np
import torch
import cv2
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from albumentations.core.composition import Compose
from lg_net.utils.utils import set_seed, flatten_omegaconf, load_obj, save_useful_info
from lg_net.datasets.get_dataset import load_augs
import Augmentor
import matplotlib.pyplot as plt
from icecream import ic


def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class Dataset:
    def __init__(
            self, root_dataset, odgt, transform: Compose, cfg: DictConfig
    ):
        """
        Prepare data for ADE20K.
        Args:
            odgt: list of image names together with labels
            cfg: other dataset configurations
        """

        # parse options
        self.cfg = cfg
        self.imgSizes = self.cfg.datamodule.imgSizes
        self.imgMaxSize = self.cfg.datamodule.imgMaxSize
        self.padding_constant = self.cfg.datamodule.padding_constant
        self.root_dataset = self.cfg.datamodule.root

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        # parse the input list
        self.max_sample = self.cfg.datamodule.max_sample
        self.start_idx = self.cfg.datamodule.start_idx
        self.end_idx = self.cfg.datamodule.end_idx
        if self.max_sample > 0:
            self.list_sample = self.list_sample[0:self.max_sample]
        if self.start_idx >= 0 and self.end_idx >= 0:  # divide file list
            self.list_sample = self.list_sample[self.start_idx:self.end_idx]

        # # mean and std
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        self.root_dataset = root_dataset
        self.transform = transform
        # down sampling rate of segment labels
        self.segm_downsampling_rate = self.cfg.datamodule.segm_downsampling_rate
        self.batch_per_gpu = self.cfg.datamodule.batch_size

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when training with batch_per_gpu > 1
        self.cur_idx = 20
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def get_item(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, str):
            this_short_size = np.random.choice([int(s) for s in self.imgSizes.split(',')])
        elif isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segment downsampling rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        f, axarr = plt.subplots(self.batch_per_gpu, 5, figsize=(30, 25))

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = (
            #         img.astype(np.float32) / 255
            # )  # Images are supposed to be float in range [0, 1]
            segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

            # note that each sample within a mini batch has different scale param
            # img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            # segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')
            #
            # # further downsample seg label, need to avoid seg label misalignment
            # segm_rounded_width = round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            # segm_rounded_height = round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            # segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            # segm_rounded.paste(segm, (0, 0))
            # segm = imresize(
            #     segm_rounded,
            #     (segm_rounded.size[0] // self.segm_downsampling_rate, segm_rounded.size[1] //
            #      self.segm_downsampling_rate), interp='nearest')
            # img = np.asarray(img)
            # segm = np.asarray(segm)
            # img = transformed['image']
            # segm = transformed['mask']

            collated_img_and_mask = [[img, segm]]
            p = Augmentor.DataPipeline(collated_img_and_mask)
            p.skew_left_right(1.0)
            # p.random_erasing(1.0, 0.12)
            # augmented_imgs = p.sample(self.batch_per_gpu)
            g = p.generator(2)
            augmented_imgs = next(g)

            transformed = self.transform(image=augmented_imgs[0][0],
                                         mask=augmented_imgs[0][1])
            transformed2 = self.transform(image=augmented_imgs[1][0],
                                          mask=augmented_imgs[1][1])

            axarr[i, 0].imshow(img)
            axarr[i, 1].imshow(transformed['image'])
            axarr[i, 2].imshow(transformed['mask'])
            axarr[i, 3].imshow(transformed2['image'])
            axarr[i, 4].imshow(transformed2['mask'])
            # put into batch arrays
            # batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            # batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        # p = Augmentor.DataPipeline(collated_img_and_mask)
        # p.flip_left_right(.5)
        # p.skew_left_right(1.0)
        # # p.random_erasing(1.0, 0.12)
        # # augmented_imgs = p.sample(self.batch_per_gpu)
        # g = p.generator(self.batch_per_gpu)
        # augmented_imgs = next(g)

        plt.show()

        output = dict()
        output['img_data'] = augmented_imgs
        # output['seg_label'] = batch_segms
        return output


def run(cfg: DictConfig, new_dir: str) -> None:
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    root_dataset = cfg.datamodule.root
    list_train = cfg.datamodule.list_train
    list_val = cfg.datamodule.list_val
    train_augs = load_augs(cfg['augmentation']['train']['augs'])
    valid_augs = load_augs(cfg['augmentation']['valid']['augs'])

    train_dataset = Dataset(root_dataset, list_train, train_augs, cfg)

    train_dataset.get_item(20)


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.general.logs_dir, exist_ok=True)
    new_dir = cfg.general.run_dir
    OmegaConf.to_yaml(cfg)
    if cfg.general.log_code:
        save_useful_info(new_dir)
    run(cfg, new_dir)


if __name__ == '__main__':
    main()
