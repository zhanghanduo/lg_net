import os
import json
import torch
from torchvision import transforms
from typing import List, Dict, Optional
from albumentations.core.composition import Compose
import cv2
import numpy as np
from PIL import Image
from lg_net.utils.utils import imresize
from torch.utils.data import Dataset
from omegaconf import DictConfig


def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


class BaseDataset(Dataset):
    def __init__(
            self, odgt, cfg: DictConfig
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

    # def img_transform(self, img):
    #     # 0-255 to 0-1
    #     img = np.float32(np.array(img)) / 255.
    #     img = img.transpose((2, 0, 1))
    #     img = self.normalize(torch.from_numpy(img.copy()))
    #     return img
    #
    # def segm_transform(self, segm):
    #     # to tensor, -1 to 149
    #     segm = torch.from_numpy(np.array(segm)).long() - 1
    #     return segm
    #
    # Round x to the nearest multiple of p and x' >= x


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, transform: Compose, cfg: DictConfig):
        super(TrainDataset, self).__init__(odgt, cfg)
        self.root_dataset = root_dataset
        self.transform = transform
        # down sampling rate of segment labels
        self.segm_downsampling_rate = self.cfg.datamodule.segm_downsampling_rate
        self.batch_per_gpu = self.cfg.datamodule.batch_size

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
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

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time.
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
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
        batch_images = np.zeros(
            (self.batch_per_gpu, 3, batch_height, batch_width))
        batch_segms = np.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate)

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

            # assert (segm.mode == "L")
            assert (img.size[0] == segm.size[0])
            assert (img.size[1] == segm.size[1])

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # # further downsample seg label, need to avoid seg label misalignment
            # segm_rounded_width = round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            # segm_rounded_height = round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            # segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            # segm_rounded.paste(segm, (0, 0))
            # segm = imresize(
            #     segm_rounded,
            #     (segm_rounded.size[0] // self.segm_downsampling_rate, segm_rounded.size[1] //
            #      self.segm_downsampling_rate), interp='nearest')

            transformed = self.transform(image=img, mask=segm)

            img = transformed['image']
            segm = transformed['mask']

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, transform: Compose, cfg: DictConfig):
        super(ValDataset, self).__init__(odgt, cfg)
        self.root_dataset = root_dataset
        self.transform = transform

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        assert (img.size[0] == segm.size[0])
        assert (img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = round2nearest_multiple(target_width, self.padding_constant)
            target_height = round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


# class TestDataset(BaseDataset):
#     def __init__(self, root_dataset, odgt, transform: Compose, cfg: DictConfig):
#         super(TestDataset, self).__init__(odgt, cfg)
#
#     def __getitem__(self, index):
#         this_record = self.list_sample[index]
#
#         # load image
#         image_path = this_record['fpath_img']
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         ori_width, ori_height = img.size
#
#         img_resized_list = []
#         for this_short_size in self.imgSizes:
#             # calculate target height and width
#             scale = min(this_short_size / float(min(ori_height, ori_width)),
#                         self.imgMaxSize / float(max(ori_height, ori_width)))
#             target_height, target_width = int(ori_height * scale), int(ori_width * scale)
#
#             # to avoid rounding in network
#             target_width = round2nearest_multiple(target_width, self.padding_constant)
#             target_height = round2nearest_multiple(target_height, self.padding_constant)
#
#             # resize images
#             img_resized = imresize(img, (target_width, target_height), interp='bilinear')
#
#             # image transform, to torch float tensor 3xHxW
#             img_resized = self.img_transform(img_resized)
#             img_resized = torch.unsqueeze(img_resized, 0)
#             img_resized_list.append(img_resized)
#
#         output = dict()
#         output['img_ori'] = np.array(img)
#         output['img_data'] = [x.contiguous() for x in img_resized_list]
#         output['info'] = this_record['fpath_img']
#         return output
#
#     def __len__(self):
#         return self.num_sample
