# -*- coding: utf-8 -*-
# Time: 2018/12/7 12:22
# Author: Dyn

from tqdm import tqdm
from config.config import *
import random
import pickle
import os
import cv2
import numpy as np


if __name__ == '__main__':
    # loading image path with label
    # dict {'0' : [...],  '1': [...], ...}

    images_path = pickle.load(open(os.sep.join([CONFIG, 'location_photos.pkl']), mode='rb'))
    location_prefix = os.sep.join([PHOTO_PATH, 'locations'])
    for key in images_path:
        for path in tqdm(images_path[key]):
            try:
                image = cv2.imread((os.path.join(location_prefix, path)))
                image = image.astype(np.float32)
            except Exception as e:
                os.remove(os.path.join(location_prefix, path))
                print(e, images_path)