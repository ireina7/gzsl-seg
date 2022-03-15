# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import os
import shutil
import src.config as config
import numpy as np
import matplotlib.pyplot as plt # type: ignore
# import dataset.voc as voc

from PIL import Image
from dataset.transform_pixel import *
from dataset.voc.gen_splits import gen_split
from dataset.voc.gen_splits import ALL_CLASSES
#from dataset.util import *
from torchvision import transforms
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from util import *
# from util.typing import *
from util.typing import torch as th



def transform_for_train(fixed_scale = 512, rotate_prob = 15, classes = None, split = None):
    """
    Options:
    1. RandomCrop
    2. CenterCrop
    3. RandomHorizontalFlip
    4. Normalize
    5. ToTensor
    6. FixedResize
    7. RandomRotate
    """
    transform_list = []
    # transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
    transform_list.append(RandomSized(fixed_scale))
    transform_list.append(RandomRotate(rotate_prob))
    transform_list.append(RandomHorizontalFlip())
    transform_list.append(Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)))
    transform_list.append(ToTensor(classes = classes, split = split))

    return transforms.Compose(transform_list)

def transform_for_test(fixed_scale = 512, rotate_prob = 15, classes = None, split = None):
    transform_list = [
        RandomSized(fixed_scale),
        RandomRotate(rotate_prob),
        Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ToTensor(classes = classes, split = split),
    ]
    return transforms.Compose(transform_list)




def dataloader_test(data_path = config.path.dataset_voc, split = "1"):
    name = "split" + split + "_"+ "train_strong"
    im_ids = []
    images = []
    categories = []

    _base_dir_image = data_path + "JPEGImages/"
    _base_dir_label = data_path + "SegmentationClass_aug/"
    with open(os.path.join(data_path, name +".txt"), "r") as f:
        lines = f.read().splitlines()
        classes = lines[0]
        lines.remove(classes)
        classes = classes.split(",")
        classes = list(map(int, classes))

    for ii, line in enumerate(lines):
        _image_name, _cat_name = line.split(" ")
        #_image_name = _image_name[1:]
        #_cat_name = _cat_name[1:]
        _image = os.path.join(_base_dir_image, _image_name)
        _cat = os.path.join(_base_dir_label, _cat_name)
        print(_image)
        print(_cat)
        assert os.path.isfile(_image)
        assert os.path.isfile(_cat)
        im_ids.append(line)
        images.append(_image)
        categories.append(_cat)

    print(classes)


def load_voc(
    data_path = config.path.dataset_voc,
    batch_size = 4,
    input_size = (512, 512),
    shuffle = True,
    num_workers = 2,
    split = None,
    mode: Mode = Mode.train_seen,
    transforming: bool = False
    ):
    """
    The main dataloader
    @param: split: 1 | 2 | 3 | 4
    """
    name = 'split{}{}{}'.format(
        split, 
        '' if mode.has(Mode.train) else '_val',
        '' if mode.has(Mode.seen ) else '_unseen',
    )
    file_path = os.path.join(data_path, name + '.txt')
    """
    It's really sad that since the code written in `gen_splits.py` 
    only considered `int` split representations, therefore we need to check many dirty things...
    """
    if split >= 0 and split <= 4:
        gen_split(split, mode = mode)
    else:
        logger.error('split range should within [0, 4], but got {}'.format(split))
    
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        classes = lines[0]
        lines.remove(classes)
        classes = classes.split(",")
        classes = list(map(int, classes))

    transform = transform_for_train(
        fixed_scale = input_size,
        rotate_prob = 15,
        classes = classes,
        split = split
    ) if transforming else \
        transform_for_test(classes = classes, split = split)

    voc_dataset: th.DataSet = VOC2012Segmentation(
        base_dir = config.path.dataset_voc,
        split = name,
        transform = transform
    )
    dataloader: th.DataLoader = DataLoader(
        voc_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = True
    )
    return dataloader
    #end load_voc



def dataloader_voc(
    data_path = config.path.dataset_voc,
    batch_size = 4,
    input_size = (512, 512),
    shuffle = True,
    num_workers = 2,
    split = None,
    mode: Mode = Mode.train_seen,
    ):
    
    dataloader = load_voc(
        data_path = data_path,
        batch_size = batch_size,
        input_size = input_size,
        shuffle = shuffle,
        num_workers = num_workers,
        split = split,
        mode = mode,
        transforming = mode.has(Mode.train)
    )
    return dataloader
    #end dataloader_voc


def dataset_voc_statistics(split_path: str) -> None:
    pass





class VOC2012Segmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(
        self,
        base_dir = config.DATA_PATH,
        split = "train",
        transform = None
        ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "voc2012/JPEGImages")
        self._cat_dir = os.path.join(self._base_dir, "voc2012/SegmentationClass")

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        logger.debug(self.split, 'self.split')
        self.transform = transform

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(_splits_dir, splt + ".txt"), "r") as f:
                lines = f.read().splitlines()
                self.classes = lines[0]
                logger.log("Reading classes: {}".format(self.classes))
                lines.remove(self.classes)
                # self.classes = list(map(lambda i: ALL_CLASSES[i], self.classes))

            for ii, line in enumerate(lines):
                # print(line)
                _image_name = line + '.jpg'
                _cat_name = line + '.png'
                #_image_name = _image_name[1:]
                #_cat_name = _cat_name[1:]
                _image = os.path.join(self._image_dir, _image_name)
                _cat = os.path.join(self._cat_dir, _cat_name)
                #print(_image)
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        logger.log("Number of images in {}: {:d}".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {"image": _img, "label": _target}
        _name = self.categories[index].split("/")[-1]
        _size = _img.size

        if self.transform is not None:
            sample = self.transform(sample)

        sample["name"] = _name
        sample["size"] = str(_size[0]) + "," + str(_size[1])
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert("RGB")).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return "VOC2012SegDataset(split = " + str(self.split) + ")"








def debug_dataset(split):
    data_loader = dataloader_voc(split = split)
    data_iter = enumerate(data_loader)
    while True:
        try:
            _, batch = data_iter.__next__()
        except StopIteration:
            data_iter = enumerate(data_loader)
            _, batch = data_iter.__next__()
        yield batch


def debug_sample(batch):
    imgs, msks = batch['image'], batch['label']
    fig, axs = plt.subplots(1, 2, figsize = (10, 3))
    axs[0].imshow(imgs[0].permute(1, 2, 0))
    axs[1].imshow(msks[0])
    #axs.set_title("test")
    #axs.grid(True)

    logger.log("Displaying image of {}".format(batch['name']))
    plt.show()
