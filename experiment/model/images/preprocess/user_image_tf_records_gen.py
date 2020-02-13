# -*- coding: utf-8 -*-
# Time: 2018/12/7 13:19
# Author: Dyn

if __name__ == '__main__':
    import pickle
    from model.images.utils.image_utils import load_image
    from tqdm import tqdm
    from config.config import *
    from model.images.utils.tf_writer import UserLocPathWriter

    images_path = pickle.load(open(USERS_PHOTOS_PATH, mode='rb'))

    cnt = 0
    with UserLocPathWriter(USERS_REC_PHOTOS) as writer:
        for key in tqdm(images_path):
            for image_path in images_path[key]:
                image = load_image(image_path, height=256, width=256)
                if image is not None:
                    writer.write(image_path, int(key))
                    cnt += 1
    print(cnt)