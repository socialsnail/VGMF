# -*- coding: utf-8 -*-
# @Time    : 2018/12/7  20:46
# @Author  : Dyn


if __name__ == '__main__':
    from config.config import *
    import pickle
    from tqdm import tqdm
    import numpy as np
    import os

    user_id_list = pickle.load(open(USERS, mode='rb'))

    user_prefix = os.sep.join([PHOTO_PATH, 'users'])
    photo_id_set = {id_: [] for id_ in user_id_list}
    cnt = 0
    for user_id in tqdm(user_id_list):
        user_id_origin = str(user_id_list[user_id])
        path = os.sep.join([user_prefix, user_id_origin])
        try:
            photos = os.listdir(path)
            # for image_path_origin in photos:
            #     p = os.path.join(user_prefix, user_id_origin, image_path_origin)
            #     if os.path.exists(os.path.join(p)):
            #         photo_id_set[user_id].append(p)
        except:
            cnt += 1
            print(user_id)
    print(cnt)

    # for key in photo_id_set.keys():
    #     photo_id_set[key] = np.array(photo_id_set[key])
    #
    # pickle.dump(photo_id_set, open(USERS_PHOTOS_PATH, mode='wb'))