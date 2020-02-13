# -*- coding: utf-8 -*-
# Time: 2018/11/6 18:16
# Author: Dyn

import imutils
import cv2
import numpy as np

vgg_mean = [103.939, 116.779, 123.68]


def load_image(image_path, mean=vgg_mean, height=224, width=224):
    # image = skimage.io.imread(image_path)
    try:
        image = cv2.imread(image_path)
        image = image.astype(np.float32)
        image = resize(image, height, width)
        # image = image[:, :, ::-1] - mean
        image = image - mean
        return image

    except Exception as e:
        print(image_path, e)
        return None


def resize(image, height, width):
    h, w = image.shape[:2]
    dw = 0
    dh = 0
    if w < h:
        image = imutils.resize(image, width=width)
        dh = int((image.shape[0] - height) / 2.0)
    else:
        image = imutils.resize(image, height=height)
        dw = int((image.shape[1] - width) / 2.0)

    h, w = image.shape[:2]
    image = image[dh:h-dh, dw:w-dw]

    return cv2.resize(image, (width, height))


if __name__ == '__main__':
    load_image('/mnt/xk/fousquareData/users/5417253/1331229515_4f58f34be4b0ad7e16377fb5.jpg')