# -*- coding: utf-8 -*-
import sys

import os
# sys.path.append('../')

import shutil
import src.config as config
import numpy as np
from PIL import Image
from dataset.transform_pixel import *
#from dataset.util import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.config import path
from src.util.typing.basic import List
from util import logger
# from util import debug
from util import Mode


'''
All classes in dataset VOC2012 (including 21 classes if background is counted)
'''
ALL_CLASSES = [
    "bg",           #  0
    "aeroplane",    #  1
    "bicycle",      #  2
    "bird",         #  3
    "boat",         #  4
    "bottle",       #  5
    "bus",          #  6
    "car",          #  7
    "cat",          #  8
    "chair",        #  9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor"     # 20
]

split = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 0
    [               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 1
    [1, 2, 3, 4, 5,                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 2
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                     16, 17, 18, 19, 20], # split 3
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15                    ]  # split 4
]



import xml.etree.ElementTree as ET
def get_class_names_of_file(
    pure_file_name, 
    xml_dir = path.join(path.dataset_voc2012, 'Annotations')
    ):
    """
    Get object names of a file, @param pure_file_name must be preprocessed!
    """
    xml_file = os.path.join(xml_dir, pure_file_name + '.xml')
    # print(xml_file)
    assert os.path.isfile(xml_file), \
        "get_class_names_of_file: file {} does not exist".format(xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objs = root.findall('object')
    ans = set()
    for obj in objs:
        currentObj = obj.find('name').text
        ans.add(currentObj)
    return list(ans)


def _gen_split(
    classes,
    voc_path = path.dataset_voc2012,
    mode: Mode = Mode.train_seen,
    file_name = 'split.txt',
    save_or_not = True
    ):
    # debug(classes, 'classes!')
    xml_dir = path.join(voc_path, 'Annotations')
    jpg_dir = path.join(voc_path, 'JPEGImages')
    seg_dir = path.join(voc_path, 'ImageSets', 'Segmentation')
    ans = {'classes': list(map(lambda cls: ALL_CLASSES.index(cls), classes)), 'files': []}
    lines = []
    session = 'train.txt' if mode.has(Mode.train) else 'val.txt'
    with open(path.join(seg_dir, session), "r") as f:
        lines = f.read().splitlines()
        logger.log("Generating splits: total {}".format(len(lines)))
    
    for line in lines:
        classes_of_file = get_class_names_of_file(line)
        for cls in classes_of_file:
            if cls in classes:
                ans['files'].append(line)
                break
    if save_or_not == True:
        file_path = config.path.join(config.path.dataset_voc, file_name)
        fp = open(file_path, 'w+')
        fp.write(', '.join(map(lambda n: str(n), ans['classes'])) + '\n')
        fp.write('\n'.join(ans['files']))
        fp.close()
        logger.log("Generated split file '{}'.".format(file_path))
    logger.log('Generated classes: {}'.format(len(ans['classes'])))
    logger.log('Generated images: {}'.format(len(ans['files'])))
    return ans


def gen_split0(
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split0.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[0])), 
        mode = mode,
        file_name = file_name
    )

def gen_split1(
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split1.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[1])), 
        mode = mode,
        file_name = file_name
    )

def gen_split2(
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split2.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[2])), 
        mode = mode,
        file_name = file_name
    )

def gen_split3(
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split3.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[3])), 
        mode = mode,
        file_name = file_name
    )

def gen_split4(
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split4.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[4])), 
        mode = mode,
        file_name = file_name
    )



'''
The public `gen_split` interface
'''
def gen_split(
    i: int,
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split.txt',
    save_or_not = True
    ):
    assert (i >= 0 and i < 5), \
        "Error while generating splits: invalid split number: {}".format(i)
    file_name = "split{}{}{}.txt".format(
        i, 
        '' if mode.has(Mode.train) else '_val',
        '' if mode.has(Mode.seen ) else '_unseen',
    )
    classes = list(map(lambda i: ALL_CLASSES[i], split[i])) if mode.has(Mode.seen)\
        else  list(map(lambda i: ALL_CLASSES[i], set(split[0]) - set(split[i])))
    return _gen_split(
        classes = classes, 
        mode = mode,
        file_name = file_name,
    )
    #end gen_split

def gen_split_by(
    split: List[int],
    voc_path = path.dataset_voc,
    mode: Mode = Mode.train_seen,
    file_name = 'split.txt',
    save_or_not = True
    ):
    # assert split valid
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split)),
        mode = mode,
        file_name = file_name,
        save_or_not = save_or_not
    )
    #end gen_split_by


