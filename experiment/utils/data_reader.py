# -*- coding: utf-8 -*-
# @Time    : 2018/6/14  15:58
# @Author  : Dyn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import gc
from config.config import *
import pickle
import json
import collections


class DataReader(object):
    def __init__(self, data_path, filter_num=10, need_restart=False,
                 valid_size=0.1, test_size=0.1):
        """Reading dataset from dick
        Args:
            data_path(str): Check-ins file path
            filter_num(int, default 10): Filter users and locations whose check-ins less than filter_num
            need_restart(bool, default False): Whether to reset the user and location id
            valid_size(float): proportion of the valid set
            test_size(float): proportion of the test set
        """
        self.filter_num = filter_num
        if not need_restart:
            self.data = self._read_with_filtering(data_path)
        else:
            self.data = self._read_from_disk(data_path)

        self.user_num = self.data.user_id.unique().shape[0]
        self.item_num = self.data.location_id.unique().shape[0]

        # dataset for train, valid and test
        self.train, self.valid, self.test = self.train_test_gen(valid_size,
                                                                test_size)
        del self.data
        gc.collect()

        # check_ins matrix of users
        self.check_ins_matrix = self._check_ins_()

    def get_check_in_matrix(self):
        """Get check ins matrix
        """
        return self.check_ins_matrix

    def _read_with_filtering(self, data_path):
        """Read dataset from disk and filter users , keep user and location in given set
        Args:
            data_path(str): Check-ins file path
        Return:
            check-in dataset frame(pandas.Dataframe)
        """
        # reading location_id and user_id to keep
        location_id = set(pickle.load(open(os.sep.join([CONFIG, 'location_id.pkl']), mode='rb')))
        user_id = set(pickle.load(open(os.sep.join([CONFIG, 'user_id.pkl']), mode='rb')))

        print('[INFO] reading dataset from disk')
        df_origin = pd.read_csv(data_path)
        df_origin = df_origin.drop_duplicates(subset=['user_id', 'location_id']).copy()
        df_origin = df_origin[(df_origin.user_id.isin(user_id)) & (df_origin.location_id.isin(location_id))]

        df_origin['user_id_origin'] = df_origin['user_id'].values
        df_origin['location_id_origin'] = df_origin['location_id'].values

        lb = LabelEncoder()

        df_origin['user_id'] = lb.fit_transform(df_origin.user_id)
        df_origin['location_id'] = lb.fit_transform(df_origin.location_id)
        df_origin.reset_index(drop=True, inplace=True)
        return df_origin

    def _read_from_disk(self, data_path):
        """Read dataset from disk and filter users and locations less than a threshold
        Args:
            data_path(str): Check-ins file path
        Return:
            check-in dataset frame(pandas.Dataframe)
        """
        print('[INFO] reading dataset from disk')
        df_origin = pd.read_csv(data_path)
        df_origin['user_id_origin'] = df_origin['user_id'].values
        df_origin['location_id_origin'] = df_origin['location_id'].values

        df = df_origin.drop_duplicates(subset=['user_id', 'location_id']).copy()

        # reading category
        categories = json.load(open('/mnt/xk/experiment/dataset/categories.json'))
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

        def cat(c):
            for index, cat in enumerate(categories_dict.values()):
                if c in cat:
                    return int(index)
            return None

        df['cat_label'] = df.categoty_id.apply(cat)
        df = df[~df.cat_label.isna()]
        df['cat_label'] = df['cat_label'].astype(np.int)

        # remove not existing category
        # categories = set(df.categoty_id.unique())
        # categories.remove(np.nan)
        # categories.remove('52f2ab2ebcbc57f1066b8b52')
        # df = df[df.categoty_id.isin(categories)]

        # remove not existing photo locations
        filter_location_id = set()
        with open(os.sep.join([CONFIG, 'no_venues_id.info'])) as f:
            for line in f.readlines():
                filter_location_id.add(line.strip('\n'))
        df = df[~df.location_id.isin(filter_location_id)]

        user_id_dict = df.groupby(by='user_id')['user_id'].count().to_dict()
        df['user_tips_cnt'] = df.user_id.map(user_id_dict)
        location_id_dict = df.groupby(by='location_id')['location_id'].count().to_dict()
        df['location_tips_cnt'] = df.location_id.map(location_id_dict)
        df_origin = df[(df.user_tips_cnt > self.filter_num) &
                       (df.location_tips_cnt > self.filter_num)].copy()
        df_origin.reset_index(drop=True, inplace=True)

        user_id_dict = df_origin.groupby(by='user_id')['user_id'].count().to_dict()
        df_origin['user_tips_cnt'] = df_origin.user_id.map(user_id_dict)
        location_id_dict = df_origin.groupby(by='location_id')['location_id'].count().to_dict()
        df_origin['location_tips_cnt'] = df_origin.location_id.map(location_id_dict)
        df_origin = df_origin[(df_origin.user_tips_cnt > self.filter_num) &
                              (df_origin.location_tips_cnt > self.filter_num)].copy()
        df_origin.reset_index(drop=True, inplace=True)

        user_id_dict = df_origin.groupby(by='user_id')['user_id'].count().to_dict()
        df_origin['user_tips_cnt'] = df_origin.user_id.map(user_id_dict)
        location_id_dict = df_origin.groupby(by='location_id')['location_id'].count().to_dict()
        df_origin['location_tips_cnt'] = df_origin.location_id.map(location_id_dict)
        df_origin = df_origin[(df_origin.user_tips_cnt > self.filter_num) &
                              (df_origin.location_tips_cnt > self.filter_num)].copy()
        df_origin.reset_index(drop=True, inplace=True)

        user_id_dict = df_origin.groupby(by='user_id')['user_id'].count().to_dict()
        df_origin['user_tips_cnt'] = df_origin.user_id.map(user_id_dict)
        location_id_dict = df_origin.groupby(by='location_id')['location_id'].count().to_dict()
        df_origin['location_tips_cnt'] = df_origin.location_id.map(location_id_dict)

        lb = LabelEncoder()
        df_origin['user_id'] = lb.fit_transform(df_origin.user_id)
        df_origin['location_id'] = lb.fit_transform(df_origin.location_id)

        pickle.dump(df_origin.user_id_origin.unique(),
                    open(os.sep.join([CONFIG, 'location_id.pkl']), mode='wb'))
        pickle.dump(df_origin.location_id_origin.unique(),
                    open(os.sep.join([CONFIG, 'user_id.pkl']), mode='wb'))
        return df_origin

    def train_test_gen(self, valid_size=0.1, test_size=0.1, seed=710):
        """Construct train, valid and test set
        Args:
            valid_size(float): proportion of the valid set
            test_size(float): proportion of the test set
            seed(int): random seed for generate dataset
        Return:
            train, valid, test dataset(pandas.Dataframe)
        """
        train_index = []
        valid_index = []
        test_index = []
        print('[INFO] gen dataset set')
        for user in self.data.user_id.unique():
            index = self.data[self.data.user_id == user].sort_values(by='time_stamp').index.values

            train, temp = train_test_split(index, test_size=valid_size + test_size, random_state=seed)
            valid, test = train_test_split(temp, test_size=test_size / (valid_size + test_size), random_state=seed)
            train_index.extend(train.tolist())
            valid_index.extend(valid.tolist())
            test_index.extend(test.tolist())

            # valid_size = max(int(len(index) * valid_size), 1)
            # test_size = max(int(len(index) * test_size), 1)
            # train_size = len(index) - valid_size - test_size
            # train = [index[x] for x in range(train_size)]
            # valid = [index[x] for x in range(train_size, train_size+valid_size)]
            # test = [index[x] for x in range(train_size+valid_size, train_size+valid_size+test_size)]
            #
            train_index.extend(train)
            valid_index.extend(valid)
            test_index.extend(test)

        return self.data.iloc[train_index].reset_index(drop=True), \
               self.data.iloc[valid_index].reset_index(drop=True), \
               self.data.iloc[test_index].reset_index(drop=True)

    def _check_ins_(self):
        """Return train set check_ins matrix
        """
        check_ins_matrix = np.zeros([self.user_num, self.item_num])

        for location_id, user_id in self.train[['location_id', 'user_id']].values:
            check_ins_matrix[user_id][location_id] += 1
        return check_ins_matrix


if __name__ == '__main__':
    from config import config
    dataset = DataReader(config.DATA_PATH, need_restart=True)