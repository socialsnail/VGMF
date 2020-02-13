# -*- coding: utf-8 -*-
# @Time    : 2018/10/12  13:32
# @Author  : Dyn


if __name__ == '__main__':
    import collections
    import json
    import os

    from config.config import *

    categories = json.load(open(os.path.sep.join([DATA_ROOT, 'categories.json'])))

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

    # reading train, valid and test dataset
    train = pd.read_csv(os.sep.join([DATA_ROOT, 'train.csv']))
    valid = pd.read_csv(os.sep.join([DATA_ROOT, 'valid.csv']))
    test = pd.read_csv(os.sep.join([DATA_ROOT, 'test.csv']))

    def cat_id2cat_num(c):
        for index, cat in enumerate(categories_dict.values()):
            if c in cat:
                return int(index)
        return None

    train['category'] = train.categoty_id.map(cat_id2cat_num)
    valid['category'] = valid.categoty_id.map(cat_id2cat_num)
    test['category'] = test.categoty_id.map(cat_id2cat_num)

    train.to_csv(os.sep.join([DATA_ROOT, 'train.csv']), index=False)
    valid.to_csv(os.sep.join([DATA_ROOT, 'valid.csv']), index=False)
    test.to_csv(os.sep.join([DATA_ROOT, 'test.csv']), index=False)