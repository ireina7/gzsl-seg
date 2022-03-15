# -*- coding: utf-8 -*-

import numpy as np
import math
import os
import sys
sys.path.append("..")
from src.config import *
from src.util import logger

def get_embeddings():
    file = ["fasttext.txt", "word2vec.txt", "all.txt"]
    embeddings = {}
    for i in file:
        dic = {}
        with open(path.join(path.dataset_semantic, i), "r") as f:
            for line in f.read().splitlines():
                name, vector = line.split(" ")
                # if name == "background":
                #    continue
                vector = vector.split(",")
                vector = list(map(float, vector))
                tmp = 0
                for x in vector:
                    tmp += x ** 2
                tmp = math.sqrt(tmp)
                dic[name] = [x / tmp for x in vector]
            embeddings[i.split(".")[0]] = dic
    return embeddings
    #end get_embeddings


def get_Ws(embeddings, strong_classes):
    file = ["fasttext", "word2vec", "all"]
    strong_len = len(strong_classes)
    Ws = {}
    for name in file:
        lenth = 300
        if name == "all":
            lenth = 600
        embedding = embeddings[name]
        strong = np.zeros([strong_len, lenth], dtype=np.float)
        weak = np.zeros([20 - strong_len, lenth], dtype=np.float)
        all = np.zeros([20, lenth], dtype=np.float)
        i, j, k = 0, 0, 0
        for class_name in embedding:
            if class_name == "background":
                continue
            if class_name in strong_classes:
                strong[i] = embedding[class_name]
                i += 1
            else:
                weak[j] = embedding[class_name]
                j += 1
            all[k] = embedding[class_name]
            k += 1
        Ws[name + "_strong"] = strong
        Ws[name + "_weak"] = weak
        Ws[name + "_all"] = all
    return Ws
    #end get_Ws



def get_Ws_split(embeddings, split):
    ALL_CLASSES = [
        "bg", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "table", "dog", "horse", "motorbike", "person", "houseplant", "sheep", "sofa", "train",
        "monitor"
    ]
    all    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    split1 = [               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    split2 = [1, 2, 3, 4, 5,                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    split3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                     16, 17, 18, 19, 20]
    split4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15                    ]
    strong_class = []
    if split == 1:
        strong_class = [ALL_CLASSES[x] for x in split1]
    elif split == 2:
        strong_class = [ALL_CLASSES[x] for x in split2]
    elif split == 3:
        strong_class = [ALL_CLASSES[x] for x in split3]
    elif split == 4:
        strong_class = [ALL_CLASSES[x] for x in split4]
    else:
        logger.error('Split should be a number within [1, 4], but got {}'.format(split))
    

    file = ["all", "fasttext", "word2vec"]
    strong_len = 15
    Ws = {}
    for name in file:
        lenth = 300
        if name == "all":
            lenth = 600
        embedding = embeddings[name]
        strong = np.zeros([strong_len, lenth], dtype=np.float)
        weak = np.zeros([20 - strong_len, lenth], dtype=np.float)

        i, j = 0, 0
        for class_name in embedding:
            if class_name == "background":
                continue
            if class_name in strong_class:
                strong[i] = embedding[class_name]
                i += 1
            else:
                weak[j] = embedding[class_name]
                j += 1
        all = np.concatenate([strong,weak])

        Ws[name + "_strong"] = strong
        Ws[name + "_weak"] = weak
        Ws[name + "_all"] = all
    return Ws





def get_embeddings_coco():
    """
    For future use of COCO dataset, currently not finished!
    """
    file = ["fasttext_coco.txt", "word2vec_coco.txt", "all_coco.txt"]
    embeddings = {}
    for i in file:
        dic = {}
        with open("model/" + i, "r") as f:
            for line in f.read().splitlines():
                name, vector = line.split(" ")
                vector = vector.split(",")
                vector = list(map(float, vector))
                # tmp = 0
                # for x in vector:
                #     tmp += x ** 2
                # tmp = math.sqrt(tmp)
                # dic[name] = [x / tmp for x in vector]
                dic[name] = vector
            embeddings[i.split("_")[0]] = dic
    return embeddings
    #end get_embeddings_coco
