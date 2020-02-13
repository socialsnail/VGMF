# -*- coding: utf-8 -*-
# @Time    : 2018/12/7  20:47
# @Author  : Dyn


if __name__ == '__main__':
    from config.config import *
    import pickle
    from tqdm import tqdm
    import numpy as np
    import os

    location_id_list = pickle.load(open(LOCATIONS, mode='rb'))

    location_prefix = os.sep.join([PHOTO_PATH, 'locations'])
    photo_id_set = {id_: [] for id_ in location_id_list}
    for location_id in tqdm(location_id_list):
        location_id_origin = location_id_list[location_id]
        path = os.sep.join([location_prefix, location_id_origin])
        try:
            photos = os.listdir(path)
            for image_path_origin in photos:
                p = os.path.join(location_prefix, location_id_origin, image_path_origin)
                if os.path.exists(os.path.join(p)):
                    photo_id_set[location_id].append(p)
        except:
            print(location_id)

    for key in photo_id_set:
        photo_id_set[key] = np.array(photo_id_set[key])
    pickle.dump(photo_id_set, open(LOCATIONS_PHOTOS_PATH, mode='wb'))
