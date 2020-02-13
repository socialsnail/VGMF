# -*- coding: utf-8 -*-
# @Time    : 2018/12/18  16:38
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
    def __init__(self, data_path, filter_num=20, need_restart=False):
        self.filter_num = filter_num
        if not need_restart:
            self.data = self._read_with_filtering(data_path)
        else:
            self.data = self._read_from_disk(data_path)

        self.user_num = self.data.user_id.unique().shape[0]
        self.item_num = self.data.location_id.unique().shape[0]

        # dataset for train, valid and test
        self.train, self.valid, self.test = self.train_test_gen()
        del self.data
        gc.collect()

        # check_ins matrix of users
        self.check_ins_matrix = self._check_ins_()

    def get_check_in_matrix(self):
        return self.check_ins_matrix

    def _read_with_filtering(self, data_path):
        # reading location_id and user_id to keep
        location_id = set(pickle.load(open(os.sep.join([CONFIG, 'location_id2.pkl']), mode='rb')))
        user_id = set(pickle.load(open(os.sep.join([CONFIG, 'user_id2.pkl']), mode='rb')))

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
        print('[INFO] reading dataset from disk')
        df_origin = pd.read_csv(data_path)
        df_origin['user_id_origin'] = df_origin['user_id'].values
        df_origin['location_id_origin'] = df_origin['location_id'].values

        df_origin = df_origin.drop_duplicates(subset=['user_id', 'location_id']).copy()

        def need_filter():
            user_id_dict = df_origin.groupby(by='user_id')['user_id'].count().to_dict()
            location_id_dict = df_origin.groupby(by='location_id')['location_id'].count().to_dict()
            return min(user_id_dict.values()) < self.filter_num or min(location_id_dict.values()) < self.filter_num

        def transforms(df_origin):
            user_id_dict = df_origin.groupby(by='user_id')['user_id'].count().to_dict()
            df_origin['user_tips_cnt'] = df_origin.user_id.map(user_id_dict)
            location_id_dict = df_origin.groupby(by='location_id')['location_id'].count().to_dict()
            df_origin['location_tips_cnt'] = df_origin.location_id.map(location_id_dict)
            df_origin = df_origin[(df_origin.user_tips_cnt >= self.filter_num) &
                                  (df_origin.location_tips_cnt >= self.filter_num)].copy()
            df_origin.reset_index(drop=True, inplace=True)
            return df_origin

        while need_filter():
            print('Filtering')
            df_origin = transforms(df_origin)

        lb = LabelEncoder()
        df_origin['user_id'] = lb.fit_transform(df_origin.user_id)
        df_origin['location_id'] = lb.fit_transform(df_origin.location_id)

        pickle.dump(df_origin.location_id_origin.unique(),
                    open(os.sep.join([CONFIG, 'location_id2.pkl']), mode='wb'))
        pickle.dump(df_origin.user_id_origin.unique(),
                    open(os.sep.join([CONFIG, 'user_id2.pkl']), mode='wb'))
        return df_origin

    def train_test_gen(self,):
        train_index = []
        valid_index = []
        test_index = []
        print('[INFO] gen dataset set')
        for user in self.data.user_id.unique():
            index = self.data[self.data.user_id == user].sort_values(by='time_stamp').index.values
            train = index[:-2]
            valid = index[-2:-1]
            test = index[-1:]
            train_index.extend(train.tolist())
            valid_index.extend(valid.tolist())
            test_index.extend(test.tolist())

        return self.data.iloc[train_index].reset_index(drop=True), \
               self.data.iloc[valid_index].reset_index(drop=True), \
               self.data.iloc[test_index].reset_index(drop=True)

    def _check_ins_(self):
        check_ins_matrix = np.zeros([self.user_num, self.item_num])

        for location_id, user_id in self.train[['location_id', 'user_id']].values:
            check_ins_matrix[user_id][location_id] += 1
        return check_ins_matrix


if __name__ == '__main__':
    DataReader(data_path=DATA_PATH, need_restart=True, filter_num=20)
