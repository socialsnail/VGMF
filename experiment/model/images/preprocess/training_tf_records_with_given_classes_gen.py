# -*- coding: utf-8 -*-
# Time: 2018/12/7 12:22
# Author: Dyn

from model.images.utils.image_utils import load_image
from tqdm import tqdm
from model.images.utils.tf_writer import ImagesPathWriter
from sklearn.model_selection import train_test_split
from config.config import *
import numpy as np
import random
import pickle
import os


if __name__ == '__main__':
    # loading image path with label
    # dict {'0' : [...],  '1': [...], ...}
    images_path_ = pickle.load(open(LOCATION_PHOTOS, mode='rb'))
    # delete classes 2 7 images
    images_path_.pop(2)
    images_path_.pop(7)
    paths = []
    random.seed(10)
    # sample images from photos to keep balance between classes
    sample_nums = 5000
    # random select files to training
    # images_path = {i: []for i in range(5)}
    # class_dict = {3:0, 4:1, 0:2, 5:3}
    # for key in images_path_.keys():
    #     if key in [3, 4, 0, 5]:
    #         images_path[class_dict[key]].extend(images_path_[key])
    #     else:
    #         images_path[4].extend(images_path_[key])

    images_path = {i: []for i in range(2)}
    for key in images_path_.keys():
        if key in [3, 4, 8, 0, 6]:
            images_path[0].extend(images_path_[key])
        else:
            images_path[1].extend(images_path_[key])

    for key in images_path.keys():
        images_path[key] = np.array(images_path[key])

    for key in images_path.keys():
        idx = [i for i in range(len(images_path[key]))]
        np.random.shuffle(idx)
        idx = random.sample(idx, sample_nums)
        paths.append(images_path[key][idx])
    # train and valid images
    train_path = []
    valid_path = []
    seed = 0
    for array in paths:
        train_, valid_ = train_test_split(array, random_state=seed, test_size=0.1)
        seed += 1
        train_path.append(train_)
        valid_path.append(valid_)

    # tf writer to saving images to disk
    train_file = CATEGORIES_CLASSIFICATION_TRAIN2_FILE
    train_path_merge = []
    train_label_merge = []
    # images writer
    for label, image_paths in enumerate(train_path):
        for image_path in image_paths:
            train_path_merge.append(image_path)
            train_label_merge.append(label)
    train_path_merge = np.asarray(train_path_merge)
    train_label_merge = np.asarray(train_label_merge)

    idx = [i for i in range(len(train_path_merge))]
    random.shuffle(idx)
    train_path_merge = train_path_merge[idx]
    train_label_merge = train_label_merge[idx]

    with ImagesPathWriter(train_file) as writer:
        for label, image_path in zip(tqdm(train_label_merge), train_path_merge):
                image = load_image(image_path, height=256, width=256)
                if image is not None:
                    writer.write(str(image_path), label)

    valid_file =CATEGORIES_CLASSIFICATION_VALID2_FILE
    valid_path_merge = []
    valid_label_merge = []
    # images writer
    for label, image_paths in enumerate(valid_path):
        for image_path in image_paths:
            valid_path_merge.append(image_path)
            valid_label_merge.append(label)
    valid_path_merge = np.asarray(valid_path_merge)
    valid_label_merge = np.asarray(valid_label_merge)

    with ImagesPathWriter(valid_file) as writer:
        for label, image_path in zip(tqdm(valid_label_merge), valid_path_merge):
                image = load_image(image_path, height=256, width=256)
                if image is not None:
                    writer.write(str(image_path), label)