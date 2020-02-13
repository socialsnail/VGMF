# -*- coding: utf-8 -*-
# @Time    : 2018/10/12  14:35
# @Author  : Dyn


if __name__ == '__main__':
    from config.config import *
    import pickle
    from tqdm import tqdm
    import numpy as np
    import os
    import pandas as pd
    from data.dataset import Dataset
    import json
    import collections
    data = Dataset()
    train = data.train
    valid = data.valid
    test = data.test

    df_origin = pd.concat([train, valid, test])
    # map categoty_id to category_label
    categories = json.load(open(CATEGORIES))
    categories_dict = {str(i): [] for i in range(10)}
    describe = {str(i): {} for i in range(10)}
    for index, cat in enumerate(categories['response']['categories']):
        describe[str(index)]['id'] = cat['id']
        describe[str(index)]['name'] = cat['name']
        categories_dict[str(index)].append(cat['id'])
        q = collections.deque()
        for c in cat['categories']:
            q.append(c)
        while len(q):
            c = q.popleft()
            categories_dict[str(index)].append(c['id'])
            for cc in c['categories']:
                q.append(cc)
    for key in categories_dict.keys():
        categories_dict[key] = set(categories_dict[key])
    def get_cat_index(c):
        for index, cat in enumerate(categories_dict.values()):
            if c in cat:
                return int(index)
        return None


    df_origin['cat_label'] = df_origin.categoty_id.map(get_cat_index)

    cat_dict = {}

    for i in range(10):
        cat_dict[i] = df_origin[df_origin.cat_label == i].location_id_origin.unique()

    location_prefix = os.sep.join([PHOTO_PATH, 'locations'])

    photo_id_set = {i: [] for i in range(10)}

    no_photos_locations = []
    for i in range(10):
        for location_id in tqdm(cat_dict[i]):
            path = os.sep.join([location_prefix, location_id])
            try:
                photos = os.listdir(path)
                photos2 = [os.sep.join([path, x]) for x in photos]
                if photos2:
                    photo_id_set[i].extend(photos2)
            except:
                no_photos_locations.append(location_id)
    for key in photo_id_set.keys():
        photo_id_set[key] = np.array(photo_id_set[key])

    pickle.dump(photo_id_set, open(LOCATION_PHOTOS, mode='wb'))
    pickle.dump(no_photos_locations, open(NO_PHOTOS_LOCATIONS, mode='wb'))